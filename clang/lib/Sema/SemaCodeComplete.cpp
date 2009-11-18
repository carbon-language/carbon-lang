//===---------------- SemaCodeComplete.cpp - Code Completion ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the code-completion semantic actions.
//
//===----------------------------------------------------------------------===//
#include "Sema.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include <list>
#include <map>
#include <vector>

using namespace clang;

namespace {
  /// \brief A container of code-completion results.
  class ResultBuilder {
  public:
    /// \brief The type of a name-lookup filter, which can be provided to the
    /// name-lookup routines to specify which declarations should be included in
    /// the result set (when it returns true) and which declarations should be
    /// filtered out (returns false).
    typedef bool (ResultBuilder::*LookupFilter)(NamedDecl *) const;
    
    typedef CodeCompleteConsumer::Result Result;
    
  private:
    /// \brief The actual results we have found.
    std::vector<Result> Results;
    
    /// \brief A record of all of the declarations we have found and placed
    /// into the result set, used to ensure that no declaration ever gets into
    /// the result set twice.
    llvm::SmallPtrSet<Decl*, 16> AllDeclsFound;
    
    /// \brief A mapping from declaration names to the declarations that have
    /// this name within a particular scope and their index within the list of
    /// results.
    typedef std::multimap<DeclarationName, 
                          std::pair<NamedDecl *, unsigned> > ShadowMap;
    
    /// \brief The semantic analysis object for which results are being 
    /// produced.
    Sema &SemaRef;
    
    /// \brief If non-NULL, a filter function used to remove any code-completion
    /// results that are not desirable.
    LookupFilter Filter;
    
    /// \brief A list of shadow maps, which is used to model name hiding at
    /// different levels of, e.g., the inheritance hierarchy.
    std::list<ShadowMap> ShadowMaps;
    
  public:
    explicit ResultBuilder(Sema &SemaRef, LookupFilter Filter = 0)
      : SemaRef(SemaRef), Filter(Filter) { }
    
    /// \brief Set the filter used for code-completion results.
    void setFilter(LookupFilter Filter) {
      this->Filter = Filter;
    }
    
    typedef std::vector<Result>::iterator iterator;
    iterator begin() { return Results.begin(); }
    iterator end() { return Results.end(); }
    
    Result *data() { return Results.empty()? 0 : &Results.front(); }
    unsigned size() const { return Results.size(); }
    bool empty() const { return Results.empty(); }
    
    /// \brief Add a new result to this result set (if it isn't already in one
    /// of the shadow maps), or replace an existing result (for, e.g., a 
    /// redeclaration).
    ///
    /// \param R the result to add (if it is unique).
    ///
    /// \param R the context in which this result will be named.
    void MaybeAddResult(Result R, DeclContext *CurContext = 0);
    
    /// \brief Enter into a new scope.
    void EnterNewScope();
    
    /// \brief Exit from the current scope.
    void ExitScope();
    
    /// \brief Ignore this declaration, if it is seen again.
    void Ignore(Decl *D) { AllDeclsFound.insert(D->getCanonicalDecl()); }

    /// \name Name lookup predicates
    ///
    /// These predicates can be passed to the name lookup functions to filter the
    /// results of name lookup. All of the predicates have the same type, so that
    /// 
    //@{
    bool IsOrdinaryName(NamedDecl *ND) const;
    bool IsNestedNameSpecifier(NamedDecl *ND) const;
    bool IsEnum(NamedDecl *ND) const;
    bool IsClassOrStruct(NamedDecl *ND) const;
    bool IsUnion(NamedDecl *ND) const;
    bool IsNamespace(NamedDecl *ND) const;
    bool IsNamespaceOrAlias(NamedDecl *ND) const;
    bool IsType(NamedDecl *ND) const;
    bool IsMember(NamedDecl *ND) const;
    //@}    
  };  
}

/// \brief Determines whether the given hidden result could be found with
/// some extra work, e.g., by qualifying the name.
///
/// \param Hidden the declaration that is hidden by the currenly \p Visible
/// declaration.
///
/// \param Visible the declaration with the same name that is already visible.
///
/// \returns true if the hidden result can be found by some mechanism,
/// false otherwise.
static bool canHiddenResultBeFound(const LangOptions &LangOpts, 
                                   NamedDecl *Hidden, NamedDecl *Visible) {
  // In C, there is no way to refer to a hidden name.
  if (!LangOpts.CPlusPlus)
    return false;
  
  DeclContext *HiddenCtx = Hidden->getDeclContext()->getLookupContext();
  
  // There is no way to qualify a name declared in a function or method.
  if (HiddenCtx->isFunctionOrMethod())
    return false;
  
  return HiddenCtx != Visible->getDeclContext()->getLookupContext();
}

/// \brief Compute the qualification required to get from the current context
/// (\p CurContext) to the target context (\p TargetContext).
///
/// \param Context the AST context in which the qualification will be used.
///
/// \param CurContext the context where an entity is being named, which is
/// typically based on the current scope.
///
/// \param TargetContext the context in which the named entity actually 
/// resides.
///
/// \returns a nested name specifier that refers into the target context, or
/// NULL if no qualification is needed.
static NestedNameSpecifier *
getRequiredQualification(ASTContext &Context,
                         DeclContext *CurContext,
                         DeclContext *TargetContext) {
  llvm::SmallVector<DeclContext *, 4> TargetParents;
  
  for (DeclContext *CommonAncestor = TargetContext;
       CommonAncestor && !CommonAncestor->Encloses(CurContext);
       CommonAncestor = CommonAncestor->getLookupParent()) {
    if (CommonAncestor->isTransparentContext() ||
        CommonAncestor->isFunctionOrMethod())
      continue;
    
    TargetParents.push_back(CommonAncestor);
  }
  
  NestedNameSpecifier *Result = 0;
  while (!TargetParents.empty()) {
    DeclContext *Parent = TargetParents.back();
    TargetParents.pop_back();
    
    if (NamespaceDecl *Namespace = dyn_cast<NamespaceDecl>(Parent))
      Result = NestedNameSpecifier::Create(Context, Result, Namespace);
    else if (TagDecl *TD = dyn_cast<TagDecl>(Parent))
      Result = NestedNameSpecifier::Create(Context, Result,
                                           false,
                                     Context.getTypeDeclType(TD).getTypePtr());
    else
      assert(Parent->isTranslationUnit());
  }  
  return Result;
}

void ResultBuilder::MaybeAddResult(Result R, DeclContext *CurContext) {
  assert(!ShadowMaps.empty() && "Must enter into a results scope");
  
  if (R.Kind != Result::RK_Declaration) {
    // For non-declaration results, just add the result.
    Results.push_back(R);
    return;
  }

  // Skip unnamed entities.
  if (!R.Declaration->getDeclName())
    return;
      
  // Look through using declarations.
  if (UsingShadowDecl *Using = dyn_cast<UsingShadowDecl>(R.Declaration))
    MaybeAddResult(Result(Using->getTargetDecl(), R.Rank, R.Qualifier),
                   CurContext);
  
  // Handle each declaration in an overload set separately.
  if (OverloadedFunctionDecl *Ovl 
        = dyn_cast<OverloadedFunctionDecl>(R.Declaration)) {
    for (OverloadedFunctionDecl::function_iterator F = Ovl->function_begin(),
         FEnd = Ovl->function_end();
         F != FEnd; ++F)
      MaybeAddResult(Result(*F, R.Rank, R.Qualifier), CurContext);
    
    return;
  }
  
  Decl *CanonDecl = R.Declaration->getCanonicalDecl();
  unsigned IDNS = CanonDecl->getIdentifierNamespace();
  
  // Friend declarations and declarations introduced due to friends are never
  // added as results.
  if (isa<FriendDecl>(CanonDecl) || 
      (IDNS & (Decl::IDNS_OrdinaryFriend | Decl::IDNS_TagFriend)))
    return;
  
  if (const IdentifierInfo *Id = R.Declaration->getIdentifier()) {
    // __va_list_tag is a freak of nature. Find it and skip it.
    if (Id->isStr("__va_list_tag") || Id->isStr("__builtin_va_list"))
      return;
    
    // Filter out names reserved for the implementation (C99 7.1.3, 
    // C++ [lib.global.names]). Users don't need to see those.
    //
    // FIXME: Add predicate for this.
    if (Id->getLength() >= 2) {
      const char *Name = Id->getNameStart();
      if (Name[0] == '_' &&
          (Name[1] == '_' || (Name[1] >= 'A' && Name[1] <= 'Z')))
        return;
    }
  }
  
  // C++ constructors are never found by name lookup.
  if (isa<CXXConstructorDecl>(CanonDecl))
    return;
  
  // Filter out any unwanted results.
  if (Filter && !(this->*Filter)(R.Declaration))
    return;
  
  ShadowMap &SMap = ShadowMaps.back();
  ShadowMap::iterator I, IEnd;
  for (llvm::tie(I, IEnd) = SMap.equal_range(R.Declaration->getDeclName());
       I != IEnd; ++I) {
    NamedDecl *ND = I->second.first;
    unsigned Index = I->second.second;
    if (ND->getCanonicalDecl() == CanonDecl) {
      // This is a redeclaration. Always pick the newer declaration.
      I->second.first = R.Declaration;
      Results[Index].Declaration = R.Declaration;
      
      // Pick the best rank of the two.
      Results[Index].Rank = std::min(Results[Index].Rank, R.Rank);
      
      // We're done.
      return;
    }
  }
  
  // This is a new declaration in this scope. However, check whether this
  // declaration name is hidden by a similarly-named declaration in an outer
  // scope.
  std::list<ShadowMap>::iterator SM, SMEnd = ShadowMaps.end();
  --SMEnd;
  for (SM = ShadowMaps.begin(); SM != SMEnd; ++SM) {
    for (llvm::tie(I, IEnd) = SM->equal_range(R.Declaration->getDeclName());
         I != IEnd; ++I) {
      // A tag declaration does not hide a non-tag declaration.
      if (I->second.first->getIdentifierNamespace() == Decl::IDNS_Tag &&
          (IDNS & (Decl::IDNS_Member | Decl::IDNS_Ordinary | 
                   Decl::IDNS_ObjCProtocol)))
        continue;
      
      // Protocols are in distinct namespaces from everything else.
      if (((I->second.first->getIdentifierNamespace() & Decl::IDNS_ObjCProtocol)
           || (IDNS & Decl::IDNS_ObjCProtocol)) &&
          I->second.first->getIdentifierNamespace() != IDNS)
        continue;
      
      // The newly-added result is hidden by an entry in the shadow map.
      if (canHiddenResultBeFound(SemaRef.getLangOptions(), R.Declaration, 
                                 I->second.first)) {
        // Note that this result was hidden.
        R.Hidden = true;
        R.QualifierIsInformative = false;
        
        if (!R.Qualifier)
          R.Qualifier = getRequiredQualification(SemaRef.Context, 
                                                 CurContext, 
                                              R.Declaration->getDeclContext());
      } else {
        // This result was hidden and cannot be found; don't bother adding
        // it.
        return;
      }
      
      break;
    }
  }
  
  // Make sure that any given declaration only shows up in the result set once.
  if (!AllDeclsFound.insert(CanonDecl))
    return;
  
  // If the filter is for nested-name-specifiers, then this result starts a
  // nested-name-specifier.
  if ((Filter == &ResultBuilder::IsNestedNameSpecifier) ||
      (Filter == &ResultBuilder::IsMember &&
       isa<CXXRecordDecl>(R.Declaration) &&
       cast<CXXRecordDecl>(R.Declaration)->isInjectedClassName()))
    R.StartsNestedNameSpecifier = true;
  
  // If this result is supposed to have an informative qualifier, add one.
  if (R.QualifierIsInformative && !R.Qualifier &&
      !R.StartsNestedNameSpecifier) {
    DeclContext *Ctx = R.Declaration->getDeclContext();
    if (NamespaceDecl *Namespace = dyn_cast<NamespaceDecl>(Ctx))
      R.Qualifier = NestedNameSpecifier::Create(SemaRef.Context, 0, Namespace);
    else if (TagDecl *Tag = dyn_cast<TagDecl>(Ctx))
      R.Qualifier = NestedNameSpecifier::Create(SemaRef.Context, 0, false, 
                             SemaRef.Context.getTypeDeclType(Tag).getTypePtr());
    else
      R.QualifierIsInformative = false;
  }
    
  // Insert this result into the set of results and into the current shadow
  // map.
  SMap.insert(std::make_pair(R.Declaration->getDeclName(),
                             std::make_pair(R.Declaration, Results.size())));
  Results.push_back(R);
}

/// \brief Enter into a new scope.
void ResultBuilder::EnterNewScope() {
  ShadowMaps.push_back(ShadowMap());
}

/// \brief Exit from the current scope.
void ResultBuilder::ExitScope() {
  ShadowMaps.pop_back();
}

/// \brief Determines whether this given declaration will be found by
/// ordinary name lookup.
bool ResultBuilder::IsOrdinaryName(NamedDecl *ND) const {
  unsigned IDNS = Decl::IDNS_Ordinary;
  if (SemaRef.getLangOptions().CPlusPlus)
    IDNS |= Decl::IDNS_Tag;
  
  return ND->getIdentifierNamespace() & IDNS;
}

/// \brief Determines whether the given declaration is suitable as the 
/// start of a C++ nested-name-specifier, e.g., a class or namespace.
bool ResultBuilder::IsNestedNameSpecifier(NamedDecl *ND) const {
  // Allow us to find class templates, too.
  if (ClassTemplateDecl *ClassTemplate = dyn_cast<ClassTemplateDecl>(ND))
    ND = ClassTemplate->getTemplatedDecl();
  
  return SemaRef.isAcceptableNestedNameSpecifier(ND);
}

/// \brief Determines whether the given declaration is an enumeration.
bool ResultBuilder::IsEnum(NamedDecl *ND) const {
  return isa<EnumDecl>(ND);
}

/// \brief Determines whether the given declaration is a class or struct.
bool ResultBuilder::IsClassOrStruct(NamedDecl *ND) const {
  // Allow us to find class templates, too.
  if (ClassTemplateDecl *ClassTemplate = dyn_cast<ClassTemplateDecl>(ND))
    ND = ClassTemplate->getTemplatedDecl();
  
  if (RecordDecl *RD = dyn_cast<RecordDecl>(ND))
    return RD->getTagKind() == TagDecl::TK_class ||
    RD->getTagKind() == TagDecl::TK_struct;
  
  return false;
}

/// \brief Determines whether the given declaration is a union.
bool ResultBuilder::IsUnion(NamedDecl *ND) const {
  // Allow us to find class templates, too.
  if (ClassTemplateDecl *ClassTemplate = dyn_cast<ClassTemplateDecl>(ND))
    ND = ClassTemplate->getTemplatedDecl();
  
  if (RecordDecl *RD = dyn_cast<RecordDecl>(ND))
    return RD->getTagKind() == TagDecl::TK_union;
  
  return false;
}

/// \brief Determines whether the given declaration is a namespace.
bool ResultBuilder::IsNamespace(NamedDecl *ND) const {
  return isa<NamespaceDecl>(ND);
}

/// \brief Determines whether the given declaration is a namespace or 
/// namespace alias.
bool ResultBuilder::IsNamespaceOrAlias(NamedDecl *ND) const {
  return isa<NamespaceDecl>(ND) || isa<NamespaceAliasDecl>(ND);
}

/// \brief Brief determines whether the given declaration is a namespace or
/// namespace alias.
bool ResultBuilder::IsType(NamedDecl *ND) const {
  return isa<TypeDecl>(ND);
}

/// \brief Since every declaration found within a class is a member that we
/// care about, always returns true. This predicate exists mostly to 
/// communicate to the result builder that we are performing a lookup for
/// member access.
bool ResultBuilder::IsMember(NamedDecl *ND) const {
  return true;
}

// Find the next outer declaration context corresponding to this scope.
static DeclContext *findOuterContext(Scope *S) {
  for (S = S->getParent(); S; S = S->getParent())
    if (S->getEntity())
      return static_cast<DeclContext *>(S->getEntity())->getPrimaryContext();
  
  return 0;
}

/// \brief Collect the results of searching for members within the given
/// declaration context.
///
/// \param Ctx the declaration context from which we will gather results.
///
/// \param Rank the rank given to results in this declaration context.
///
/// \param Visited the set of declaration contexts that have already been
/// visited. Declaration contexts will only be visited once.
///
/// \param Results the result set that will be extended with any results
/// found within this declaration context (and, for a C++ class, its bases).
///
/// \param InBaseClass whether we are in a base class.
///
/// \returns the next higher rank value, after considering all of the
/// names within this declaration context.
static unsigned CollectMemberLookupResults(DeclContext *Ctx, 
                                           unsigned Rank,
                                           DeclContext *CurContext,
                                 llvm::SmallPtrSet<DeclContext *, 16> &Visited,
                                           ResultBuilder &Results,
                                           bool InBaseClass = false) {
  // Make sure we don't visit the same context twice.
  if (!Visited.insert(Ctx->getPrimaryContext()))
    return Rank;
  
  // Enumerate all of the results in this context.
  typedef CodeCompleteConsumer::Result Result;
  Results.EnterNewScope();
  for (DeclContext *CurCtx = Ctx->getPrimaryContext(); CurCtx; 
       CurCtx = CurCtx->getNextContext()) {
    for (DeclContext::decl_iterator D = CurCtx->decls_begin(), 
                                 DEnd = CurCtx->decls_end();
         D != DEnd; ++D) {
      if (NamedDecl *ND = dyn_cast<NamedDecl>(*D))
        Results.MaybeAddResult(Result(ND, Rank, 0, InBaseClass), CurContext);
      
      // Visit transparent contexts inside this context.
      if (DeclContext *InnerCtx = dyn_cast<DeclContext>(*D)) {
        if (InnerCtx->isTransparentContext())
          CollectMemberLookupResults(InnerCtx, Rank, CurContext, Visited,
                                     Results, InBaseClass);
      }
    }
  }
  
  // Traverse the contexts of inherited classes.
  if (CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(Ctx)) {
    for (CXXRecordDecl::base_class_iterator B = Record->bases_begin(),
                                         BEnd = Record->bases_end();
         B != BEnd; ++B) {
      QualType BaseType = B->getType();
      
      // Don't look into dependent bases, because name lookup can't look
      // there anyway.
      if (BaseType->isDependentType())
        continue;
      
      const RecordType *Record = BaseType->getAs<RecordType>();
      if (!Record)
        continue;
      
      // FIXME: It would be nice to be able to determine whether referencing
      // a particular member would be ambiguous. For example, given
      //
      //   struct A { int member; };
      //   struct B { int member; };
      //   struct C : A, B { };
      //
      //   void f(C *c) { c->### }
      // accessing 'member' would result in an ambiguity. However, code
      // completion could be smart enough to qualify the member with the
      // base class, e.g.,
      //
      //   c->B::member
      //
      // or
      //
      //   c->A::member
      
      // Collect results from this base class (and its bases).
      CollectMemberLookupResults(Record->getDecl(), Rank, CurContext, Visited,
                                 Results, /*InBaseClass=*/true);
    }
  }
  
  // FIXME: Look into base classes in Objective-C!
  
  Results.ExitScope();
  return Rank + 1;
}

/// \brief Collect the results of searching for members within the given
/// declaration context.
///
/// \param Ctx the declaration context from which we will gather results.
///
/// \param InitialRank the initial rank given to results in this declaration
/// context. Larger rank values will be used for, e.g., members found in
/// base classes.
///
/// \param Results the result set that will be extended with any results
/// found within this declaration context (and, for a C++ class, its bases).
///
/// \returns the next higher rank value, after considering all of the
/// names within this declaration context.
static unsigned CollectMemberLookupResults(DeclContext *Ctx, 
                                           unsigned InitialRank, 
                                           DeclContext *CurContext,
                                           ResultBuilder &Results) {
  llvm::SmallPtrSet<DeclContext *, 16> Visited;
  return CollectMemberLookupResults(Ctx, InitialRank, CurContext, Visited, 
                                    Results);
}

/// \brief Collect the results of searching for declarations within the given
/// scope and its parent scopes.
///
/// \param S the scope in which we will start looking for declarations.
///
/// \param InitialRank the initial rank given to results in this scope.
/// Larger rank values will be used for results found in parent scopes.
///
/// \param CurContext the context from which lookup results will be found.
///
/// \param Results the builder object that will receive each result.
static unsigned CollectLookupResults(Scope *S, 
                                     TranslationUnitDecl *TranslationUnit,
                                     unsigned InitialRank,
                                     DeclContext *CurContext,
                                     ResultBuilder &Results) {
  if (!S)
    return InitialRank;
  
  // FIXME: Using directives!
  
  unsigned NextRank = InitialRank;
  Results.EnterNewScope();
  if (S->getEntity() && 
      !((DeclContext *)S->getEntity())->isFunctionOrMethod()) {
    // Look into this scope's declaration context, along with any of its
    // parent lookup contexts (e.g., enclosing classes), up to the point
    // where we hit the context stored in the next outer scope.
    DeclContext *Ctx = (DeclContext *)S->getEntity();
    DeclContext *OuterCtx = findOuterContext(S);
    
    for (; Ctx && Ctx->getPrimaryContext() != OuterCtx;
         Ctx = Ctx->getLookupParent()) {
      if (Ctx->isFunctionOrMethod())
        continue;
      
      NextRank = CollectMemberLookupResults(Ctx, NextRank + 1, CurContext,
                                            Results);
    }
  } else if (!S->getParent()) {
    // Look into the translation unit scope. We walk through the translation
    // unit's declaration context, because the Scope itself won't have all of
    // the declarations if we loaded a precompiled header.
    // FIXME: We would like the translation unit's Scope object to point to the
    // translation unit, so we don't need this special "if" branch. However,
    // doing so would force the normal C++ name-lookup code to look into the
    // translation unit decl when the IdentifierInfo chains would suffice. 
    // Once we fix that problem (which is part of a more general "don't look
    // in DeclContexts unless we have to" optimization), we can eliminate the
    // TranslationUnit parameter entirely.
    NextRank = CollectMemberLookupResults(TranslationUnit, NextRank + 1, 
                                          CurContext, Results);
  } else {
    // Walk through the declarations in this Scope.
    for (Scope::decl_iterator D = S->decl_begin(), DEnd = S->decl_end();
         D != DEnd; ++D) {
      if (NamedDecl *ND = dyn_cast<NamedDecl>((Decl *)((*D).get())))
        Results.MaybeAddResult(CodeCompleteConsumer::Result(ND, NextRank),
                               CurContext);        
    }
    
    NextRank = NextRank + 1;
  }
  
  // Lookup names in the parent scope.
  NextRank = CollectLookupResults(S->getParent(), TranslationUnit, NextRank, 
                                  CurContext, Results);
  Results.ExitScope();
  
  return NextRank;
}

/// \brief Add type specifiers for the current language as keyword results.
static void AddTypeSpecifierResults(const LangOptions &LangOpts, unsigned Rank, 
                                    ResultBuilder &Results) {
  typedef CodeCompleteConsumer::Result Result;
  Results.MaybeAddResult(Result("short", Rank));
  Results.MaybeAddResult(Result("long", Rank));
  Results.MaybeAddResult(Result("signed", Rank));
  Results.MaybeAddResult(Result("unsigned", Rank));
  Results.MaybeAddResult(Result("void", Rank));
  Results.MaybeAddResult(Result("char", Rank));
  Results.MaybeAddResult(Result("int", Rank));
  Results.MaybeAddResult(Result("float", Rank));
  Results.MaybeAddResult(Result("double", Rank));
  Results.MaybeAddResult(Result("enum", Rank));
  Results.MaybeAddResult(Result("struct", Rank));
  Results.MaybeAddResult(Result("union", Rank));
  
  if (LangOpts.C99) {
    // C99-specific
    Results.MaybeAddResult(Result("_Complex", Rank));
    Results.MaybeAddResult(Result("_Imaginary", Rank));
    Results.MaybeAddResult(Result("_Bool", Rank));
  }
  
  if (LangOpts.CPlusPlus) {
    // C++-specific
    Results.MaybeAddResult(Result("bool", Rank));
    Results.MaybeAddResult(Result("class", Rank));
    Results.MaybeAddResult(Result("typename", Rank));
    Results.MaybeAddResult(Result("wchar_t", Rank));
    
    if (LangOpts.CPlusPlus0x) {
      Results.MaybeAddResult(Result("char16_t", Rank));
      Results.MaybeAddResult(Result("char32_t", Rank));
      Results.MaybeAddResult(Result("decltype", Rank));
    }
  }
  
  // GNU extensions
  if (LangOpts.GNUMode) {
    // FIXME: Enable when we actually support decimal floating point.
    //    Results.MaybeAddResult(Result("_Decimal32", Rank));
    //    Results.MaybeAddResult(Result("_Decimal64", Rank));
    //    Results.MaybeAddResult(Result("_Decimal128", Rank));
    Results.MaybeAddResult(Result("typeof", Rank));
  }
}

/// \brief Add function parameter chunks to the given code completion string.
static void AddFunctionParameterChunks(ASTContext &Context,
                                       FunctionDecl *Function,
                                       CodeCompletionString *Result) {
  typedef CodeCompletionString::Chunk Chunk;
  
  CodeCompletionString *CCStr = Result;
  
  for (unsigned P = 0, N = Function->getNumParams(); P != N; ++P) {
    ParmVarDecl *Param = Function->getParamDecl(P);
    
    if (Param->hasDefaultArg()) {
      // When we see an optional default argument, put that argument and
      // the remaining default arguments into a new, optional string.
      CodeCompletionString *Opt = new CodeCompletionString;
      CCStr->AddOptionalChunk(std::auto_ptr<CodeCompletionString>(Opt));
      CCStr = Opt;
    }
    
    if (P != 0)
      CCStr->AddChunk(Chunk(CodeCompletionString::CK_Comma));
    
    // Format the placeholder string.
    std::string PlaceholderStr;
    if (Param->getIdentifier())
      PlaceholderStr = Param->getIdentifier()->getName();
    
    Param->getType().getAsStringInternal(PlaceholderStr, 
                                         Context.PrintingPolicy);
    
    // Add the placeholder string.
    CCStr->AddPlaceholderChunk(PlaceholderStr.c_str());
  }
  
  if (const FunctionProtoType *Proto 
        = Function->getType()->getAs<FunctionProtoType>())
    if (Proto->isVariadic())
      CCStr->AddPlaceholderChunk(", ...");
}

/// \brief Add template parameter chunks to the given code completion string.
static void AddTemplateParameterChunks(ASTContext &Context,
                                       TemplateDecl *Template,
                                       CodeCompletionString *Result,
                                       unsigned MaxParameters = 0) {
  typedef CodeCompletionString::Chunk Chunk;
  
  CodeCompletionString *CCStr = Result;
  bool FirstParameter = true;
  
  TemplateParameterList *Params = Template->getTemplateParameters();
  TemplateParameterList::iterator PEnd = Params->end();
  if (MaxParameters)
    PEnd = Params->begin() + MaxParameters;
  for (TemplateParameterList::iterator P = Params->begin(); P != PEnd; ++P) {
    bool HasDefaultArg = false;
    std::string PlaceholderStr;
    if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(*P)) {
      if (TTP->wasDeclaredWithTypename())
        PlaceholderStr = "typename";
      else
        PlaceholderStr = "class";
      
      if (TTP->getIdentifier()) {
        PlaceholderStr += ' ';
        PlaceholderStr += TTP->getIdentifier()->getName();
      }
      
      HasDefaultArg = TTP->hasDefaultArgument();
    } else if (NonTypeTemplateParmDecl *NTTP 
               = dyn_cast<NonTypeTemplateParmDecl>(*P)) {
      if (NTTP->getIdentifier())
        PlaceholderStr = NTTP->getIdentifier()->getName();
      NTTP->getType().getAsStringInternal(PlaceholderStr, 
                                          Context.PrintingPolicy);
      HasDefaultArg = NTTP->hasDefaultArgument();
    } else {
      assert(isa<TemplateTemplateParmDecl>(*P));
      TemplateTemplateParmDecl *TTP = cast<TemplateTemplateParmDecl>(*P);
      
      // Since putting the template argument list into the placeholder would
      // be very, very long, we just use an abbreviation.
      PlaceholderStr = "template<...> class";
      if (TTP->getIdentifier()) {
        PlaceholderStr += ' ';
        PlaceholderStr += TTP->getIdentifier()->getName();
      }
      
      HasDefaultArg = TTP->hasDefaultArgument();
    }
    
    if (HasDefaultArg) {
      // When we see an optional default argument, put that argument and
      // the remaining default arguments into a new, optional string.
      CodeCompletionString *Opt = new CodeCompletionString;
      CCStr->AddOptionalChunk(std::auto_ptr<CodeCompletionString>(Opt));
      CCStr = Opt;
    }
    
    if (FirstParameter)
      FirstParameter = false;
    else
      CCStr->AddChunk(Chunk(CodeCompletionString::CK_Comma));
    
    // Add the placeholder string.
    CCStr->AddPlaceholderChunk(PlaceholderStr.c_str());
  }    
}

/// \brief Add a qualifier to the given code-completion string, if the
/// provided nested-name-specifier is non-NULL.
void AddQualifierToCompletionString(CodeCompletionString *Result, 
                                    NestedNameSpecifier *Qualifier, 
                                    bool QualifierIsInformative,
                                    ASTContext &Context) {
  if (!Qualifier)
    return;
  
  std::string PrintedNNS;
  {
    llvm::raw_string_ostream OS(PrintedNNS);
    Qualifier->print(OS, Context.PrintingPolicy);
  }
  if (QualifierIsInformative)
    Result->AddInformativeChunk(PrintedNNS.c_str());
  else
    Result->AddTextChunk(PrintedNNS.c_str());
}

/// \brief If possible, create a new code completion string for the given
/// result.
///
/// \returns Either a new, heap-allocated code completion string describing
/// how to use this result, or NULL to indicate that the string or name of the
/// result is all that is needed.
CodeCompletionString *
CodeCompleteConsumer::Result::CreateCodeCompletionString(Sema &S) {
  typedef CodeCompletionString::Chunk Chunk;
  
  if (Kind == RK_Keyword)
    return 0;
  
  if (Kind == RK_Macro) {
    MacroInfo *MI = S.PP.getMacroInfo(Macro);
    if (!MI || !MI->isFunctionLike())
      return 0;
    
    // Format a function-like macro with placeholders for the arguments.
    CodeCompletionString *Result = new CodeCompletionString;
    Result->AddTypedTextChunk(Macro->getName().str().c_str());
    Result->AddChunk(Chunk(CodeCompletionString::CK_LeftParen));
    for (MacroInfo::arg_iterator A = MI->arg_begin(), AEnd = MI->arg_end();
         A != AEnd; ++A) {
      if (A != MI->arg_begin())
        Result->AddChunk(Chunk(CodeCompletionString::CK_Comma));
      
      if (!MI->isVariadic() || A != AEnd - 1) {
        // Non-variadic argument.
        Result->AddPlaceholderChunk((*A)->getName().str().c_str());
        continue;
      }
      
      // Variadic argument; cope with the different between GNU and C99
      // variadic macros, providing a single placeholder for the rest of the
      // arguments.
      if ((*A)->isStr("__VA_ARGS__"))
        Result->AddPlaceholderChunk("...");
      else {
        std::string Arg = (*A)->getName();
        Arg += "...";
        Result->AddPlaceholderChunk(Arg.c_str());
      }
    }
    Result->AddChunk(Chunk(CodeCompletionString::CK_RightParen));
    return Result;
  }
  
  assert(Kind == RK_Declaration && "Missed a macro kind?");
  NamedDecl *ND = Declaration;
  
  if (StartsNestedNameSpecifier) {
    CodeCompletionString *Result = new CodeCompletionString;
    Result->AddTypedTextChunk(ND->getNameAsString().c_str());
    Result->AddTextChunk("::");
    return Result;
  }
  
  if (FunctionDecl *Function = dyn_cast<FunctionDecl>(ND)) {
    CodeCompletionString *Result = new CodeCompletionString;
    AddQualifierToCompletionString(Result, Qualifier, QualifierIsInformative, 
                                   S.Context);
    Result->AddTypedTextChunk(Function->getNameAsString().c_str());
    Result->AddChunk(Chunk(CodeCompletionString::CK_LeftParen));
    AddFunctionParameterChunks(S.Context, Function, Result);
    Result->AddChunk(Chunk(CodeCompletionString::CK_RightParen));
    return Result;
  }
  
  if (FunctionTemplateDecl *FunTmpl = dyn_cast<FunctionTemplateDecl>(ND)) {
    CodeCompletionString *Result = new CodeCompletionString;
    AddQualifierToCompletionString(Result, Qualifier, QualifierIsInformative, 
                                   S.Context);
    FunctionDecl *Function = FunTmpl->getTemplatedDecl();
    Result->AddTypedTextChunk(Function->getNameAsString().c_str());
    
    // Figure out which template parameters are deduced (or have default
    // arguments).
    llvm::SmallVector<bool, 16> Deduced;
    S.MarkDeducedTemplateParameters(FunTmpl, Deduced);
    unsigned LastDeducibleArgument;
    for (LastDeducibleArgument = Deduced.size(); LastDeducibleArgument > 0;
         --LastDeducibleArgument) {
      if (!Deduced[LastDeducibleArgument - 1]) {
        // C++0x: Figure out if the template argument has a default. If so,
        // the user doesn't need to type this argument.
        // FIXME: We need to abstract template parameters better!
        bool HasDefaultArg = false;
        NamedDecl *Param = FunTmpl->getTemplateParameters()->getParam(
                                                                      LastDeducibleArgument - 1);
        if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(Param))
          HasDefaultArg = TTP->hasDefaultArgument();
        else if (NonTypeTemplateParmDecl *NTTP 
                 = dyn_cast<NonTypeTemplateParmDecl>(Param))
          HasDefaultArg = NTTP->hasDefaultArgument();
        else {
          assert(isa<TemplateTemplateParmDecl>(Param));
          HasDefaultArg 
            = cast<TemplateTemplateParmDecl>(Param)->hasDefaultArgument();
        }
        
        if (!HasDefaultArg)
          break;
      }
    }
    
    if (LastDeducibleArgument) {
      // Some of the function template arguments cannot be deduced from a
      // function call, so we introduce an explicit template argument list
      // containing all of the arguments up to the first deducible argument.
      Result->AddChunk(Chunk(CodeCompletionString::CK_LeftAngle));
      AddTemplateParameterChunks(S.Context, FunTmpl, Result, 
                                 LastDeducibleArgument);
      Result->AddChunk(Chunk(CodeCompletionString::CK_RightAngle));
    }
    
    // Add the function parameters
    Result->AddChunk(Chunk(CodeCompletionString::CK_LeftParen));
    AddFunctionParameterChunks(S.Context, Function, Result);
    Result->AddChunk(Chunk(CodeCompletionString::CK_RightParen));
    return Result;
  }
  
  if (TemplateDecl *Template = dyn_cast<TemplateDecl>(ND)) {
    CodeCompletionString *Result = new CodeCompletionString;
    AddQualifierToCompletionString(Result, Qualifier, QualifierIsInformative, 
                                   S.Context);
    Result->AddTypedTextChunk(Template->getNameAsString().c_str());
    Result->AddChunk(Chunk(CodeCompletionString::CK_LeftAngle));
    AddTemplateParameterChunks(S.Context, Template, Result);
    Result->AddChunk(Chunk(CodeCompletionString::CK_RightAngle));
    return Result;
  }
  
  if (ObjCMethodDecl *Method = dyn_cast<ObjCMethodDecl>(ND)) {
    CodeCompletionString *Result = new CodeCompletionString;
    Selector Sel = Method->getSelector();
    if (Sel.isUnarySelector()) {
      Result->AddTypedTextChunk(Sel.getIdentifierInfoForSlot(0)->getName());
      return Result;
    }

    Result->AddTypedTextChunk(
          Sel.getIdentifierInfoForSlot(0)->getName().str() + std::string(":"));
    unsigned Idx = 0;
    for (ObjCMethodDecl::param_iterator P = Method->param_begin(),
                                     PEnd = Method->param_end();
         P != PEnd; (void)++P, ++Idx) {
      if (Idx > 0) {
        std::string Keyword = " ";
        if (IdentifierInfo *II = Sel.getIdentifierInfoForSlot(Idx))
          Keyword += II->getName().str();
        Keyword += ":";
        Result->AddTextChunk(Keyword);
      }

      std::string Arg;
      (*P)->getType().getAsStringInternal(Arg, S.Context.PrintingPolicy);
      Arg = "(" + Arg + ")";
      if (IdentifierInfo *II = (*P)->getIdentifier())
        Arg += II->getName().str();
      Result->AddPlaceholderChunk(Arg);
    }

    return Result;
  }

  if (Qualifier) {
    CodeCompletionString *Result = new CodeCompletionString;
    AddQualifierToCompletionString(Result, Qualifier, QualifierIsInformative, 
                                   S.Context);
    Result->AddTypedTextChunk(ND->getNameAsString().c_str());
    return Result;
  }
  
  return 0;
}

CodeCompletionString *
CodeCompleteConsumer::OverloadCandidate::CreateSignatureString(
                                                          unsigned CurrentArg,
                                                               Sema &S) const {
  typedef CodeCompletionString::Chunk Chunk;
  
  CodeCompletionString *Result = new CodeCompletionString;
  FunctionDecl *FDecl = getFunction();
  const FunctionProtoType *Proto 
    = dyn_cast<FunctionProtoType>(getFunctionType());
  if (!FDecl && !Proto) {
    // Function without a prototype. Just give the return type and a 
    // highlighted ellipsis.
    const FunctionType *FT = getFunctionType();
    Result->AddTextChunk(
            FT->getResultType().getAsString(S.Context.PrintingPolicy).c_str());
    Result->AddChunk(Chunk(CodeCompletionString::CK_LeftParen));
    Result->AddChunk(Chunk(CodeCompletionString::CK_CurrentParameter, "..."));
    Result->AddChunk(Chunk(CodeCompletionString::CK_RightParen));
    return Result;
  }
  
  if (FDecl)
    Result->AddTextChunk(FDecl->getNameAsString().c_str());    
  else
    Result->AddTextChunk(
         Proto->getResultType().getAsString(S.Context.PrintingPolicy).c_str());
  
  Result->AddChunk(Chunk(CodeCompletionString::CK_LeftParen));
  unsigned NumParams = FDecl? FDecl->getNumParams() : Proto->getNumArgs();
  for (unsigned I = 0; I != NumParams; ++I) {
    if (I)
      Result->AddChunk(Chunk(CodeCompletionString::CK_Comma));
    
    std::string ArgString;
    QualType ArgType;
    
    if (FDecl) {
      ArgString = FDecl->getParamDecl(I)->getNameAsString();
      ArgType = FDecl->getParamDecl(I)->getOriginalType();
    } else {
      ArgType = Proto->getArgType(I);
    }
    
    ArgType.getAsStringInternal(ArgString, S.Context.PrintingPolicy);
    
    if (I == CurrentArg)
      Result->AddChunk(Chunk(CodeCompletionString::CK_CurrentParameter, 
                             ArgString.c_str()));
    else
      Result->AddTextChunk(ArgString.c_str());
  }
  
  if (Proto && Proto->isVariadic()) {
    Result->AddChunk(Chunk(CodeCompletionString::CK_Comma));
    if (CurrentArg < NumParams)
      Result->AddTextChunk("...");
    else
      Result->AddChunk(Chunk(CodeCompletionString::CK_CurrentParameter, "..."));
  }
  Result->AddChunk(Chunk(CodeCompletionString::CK_RightParen));
  
  return Result;
}

namespace {
  struct SortCodeCompleteResult {
    typedef CodeCompleteConsumer::Result Result;
    
    bool isEarlierDeclarationName(DeclarationName X, DeclarationName Y) const {
      if (!X.getObjCSelector().isNull() && !Y.getObjCSelector().isNull()) {
        // Consider all selector kinds to be equivalent.
      } else if (X.getNameKind() != Y.getNameKind())
        return X.getNameKind() < Y.getNameKind();
      
      return llvm::LowercaseString(X.getAsString()) 
        < llvm::LowercaseString(Y.getAsString());
    }
    
    bool operator()(const Result &X, const Result &Y) const {
      // Sort first by rank.
      if (X.Rank < Y.Rank)
        return true;
      else if (X.Rank > Y.Rank)
        return false;
      
      // Result kinds are ordered by decreasing importance.
      if (X.Kind < Y.Kind)
        return true;
      else if (X.Kind > Y.Kind)
        return false;
      
      // Non-hidden names precede hidden names.
      if (X.Hidden != Y.Hidden)
        return !X.Hidden;
      
      // Non-nested-name-specifiers precede nested-name-specifiers.
      if (X.StartsNestedNameSpecifier != Y.StartsNestedNameSpecifier)
        return !X.StartsNestedNameSpecifier;
      
      // Ordering depends on the kind of result.
      switch (X.Kind) {
        case Result::RK_Declaration:
          // Order based on the declaration names.
          return isEarlierDeclarationName(X.Declaration->getDeclName(),
                                          Y.Declaration->getDeclName());
          
        case Result::RK_Keyword:
          return strcmp(X.Keyword, Y.Keyword) < 0;
          
        case Result::RK_Macro:
          return llvm::LowercaseString(X.Macro->getName()) < 
                   llvm::LowercaseString(Y.Macro->getName());
      }
      
      // Silence GCC warning.
      return false;
    }
  };
}

static void AddMacroResults(Preprocessor &PP, unsigned Rank, 
                            ResultBuilder &Results) {
  Results.EnterNewScope();
  for (Preprocessor::macro_iterator M = PP.macro_begin(), 
                                 MEnd = PP.macro_end();
       M != MEnd; ++M)
    Results.MaybeAddResult(CodeCompleteConsumer::Result(M->first, Rank));
  Results.ExitScope();
}

static void HandleCodeCompleteResults(Sema *S,
                                      CodeCompleteConsumer *CodeCompleter,
                                     CodeCompleteConsumer::Result *Results,
                                     unsigned NumResults) {
  // Sort the results by rank/kind/etc.
  std::stable_sort(Results, Results + NumResults, SortCodeCompleteResult());

  if (CodeCompleter)
    CodeCompleter->ProcessCodeCompleteResults(*S, Results, NumResults);
}

void Sema::CodeCompleteOrdinaryName(Scope *S) {
  ResultBuilder Results(*this, &ResultBuilder::IsOrdinaryName);
  unsigned NextRank = CollectLookupResults(S, Context.getTranslationUnitDecl(), 
                                           0, CurContext, Results);
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, NextRank, Results);
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

static void AddObjCProperties(ObjCContainerDecl *Container, 
                              DeclContext *CurContext,
                              ResultBuilder &Results) {
  typedef CodeCompleteConsumer::Result Result;

  // Add properties in this container.
  for (ObjCContainerDecl::prop_iterator P = Container->prop_begin(),
                                     PEnd = Container->prop_end();
       P != PEnd;
       ++P)
    Results.MaybeAddResult(Result(*P, 0), CurContext);
  
  // Add properties in referenced protocols.
  if (ObjCProtocolDecl *Protocol = dyn_cast<ObjCProtocolDecl>(Container)) {
    for (ObjCProtocolDecl::protocol_iterator P = Protocol->protocol_begin(),
                                          PEnd = Protocol->protocol_end();
         P != PEnd; ++P)
      AddObjCProperties(*P, CurContext, Results);
  } else if (ObjCInterfaceDecl *IFace = dyn_cast<ObjCInterfaceDecl>(Container)){
    // Look through categories.
    for (ObjCCategoryDecl *Category = IFace->getCategoryList();
         Category; Category = Category->getNextClassCategory())
      AddObjCProperties(Category, CurContext, Results);
    
    // Look through protocols.
    for (ObjCInterfaceDecl::protocol_iterator I = IFace->protocol_begin(),
                                              E = IFace->protocol_end(); 
         I != E; ++I)
      AddObjCProperties(*I, CurContext, Results);
    
    // Look in the superclass.
    if (IFace->getSuperClass())
      AddObjCProperties(IFace->getSuperClass(), CurContext, Results);
  } else if (const ObjCCategoryDecl *Category
                                    = dyn_cast<ObjCCategoryDecl>(Container)) {
    // Look through protocols.
    for (ObjCInterfaceDecl::protocol_iterator P = Category->protocol_begin(),
                                           PEnd = Category->protocol_end(); 
         P != PEnd; ++P)
      AddObjCProperties(*P, CurContext, Results);
  }
}

void Sema::CodeCompleteMemberReferenceExpr(Scope *S, ExprTy *BaseE,
                                           SourceLocation OpLoc,
                                           bool IsArrow) {
  if (!BaseE || !CodeCompleter)
    return;
  
  typedef CodeCompleteConsumer::Result Result;
  
  Expr *Base = static_cast<Expr *>(BaseE);
  QualType BaseType = Base->getType();

  if (IsArrow) {
    if (const PointerType *Ptr = BaseType->getAs<PointerType>())
      BaseType = Ptr->getPointeeType();
    else if (BaseType->isObjCObjectPointerType())
    /*Do nothing*/ ;
    else
      return;
  }
  
  ResultBuilder Results(*this, &ResultBuilder::IsMember);
  unsigned NextRank = 0;

  Results.EnterNewScope();
  if (const RecordType *Record = BaseType->getAs<RecordType>()) {
    // Access to a C/C++ class, struct, or union.
    NextRank = CollectMemberLookupResults(Record->getDecl(), NextRank,
                                          Record->getDecl(), Results);

    if (getLangOptions().CPlusPlus) {
      if (!Results.empty()) {
        // The "template" keyword can follow "->" or "." in the grammar.
        // However, we only want to suggest the template keyword if something
        // is dependent.
        bool IsDependent = BaseType->isDependentType();
        if (!IsDependent) {
          for (Scope *DepScope = S; DepScope; DepScope = DepScope->getParent())
            if (DeclContext *Ctx = (DeclContext *)DepScope->getEntity()) {
              IsDependent = Ctx->isDependentContext();
              break;
            }
        }

        if (IsDependent)
          Results.MaybeAddResult(Result("template", NextRank++));
      }

      // We could have the start of a nested-name-specifier. Add those
      // results as well.
      Results.setFilter(&ResultBuilder::IsNestedNameSpecifier);
      CollectLookupResults(S, Context.getTranslationUnitDecl(), NextRank,
                           CurContext, Results);
    }
  } else if (!IsArrow && BaseType->getAsObjCInterfacePointerType()) {
    // Objective-C property reference.
    
    // Add property results based on our interface.
    const ObjCObjectPointerType *ObjCPtr
      = BaseType->getAsObjCInterfacePointerType();
    assert(ObjCPtr && "Non-NULL pointer guaranteed above!");
    AddObjCProperties(ObjCPtr->getInterfaceDecl(), CurContext, Results);
    
    // Add properties from the protocols in a qualified interface.
    for (ObjCObjectPointerType::qual_iterator I = ObjCPtr->qual_begin(),
                                              E = ObjCPtr->qual_end();
         I != E; ++I)
      AddObjCProperties(*I, CurContext, Results);
    
    // FIXME: We could (should?) also look for "implicit" properties, identified
    // only by the presence of nullary and unary selectors.
  } else if ((IsArrow && BaseType->isObjCObjectPointerType()) ||
             (!IsArrow && BaseType->isObjCInterfaceType())) {
    // Objective-C instance variable access.
    ObjCInterfaceDecl *Class = 0;
    if (const ObjCObjectPointerType *ObjCPtr
                                    = BaseType->getAs<ObjCObjectPointerType>())
      Class = ObjCPtr->getInterfaceDecl();
    else
      Class = BaseType->getAs<ObjCInterfaceType>()->getDecl();
    
    // Add all ivars from this class and its superclasses.
    for (; Class; Class = Class->getSuperClass()) {
      for (ObjCInterfaceDecl::ivar_iterator IVar = Class->ivar_begin(), 
                                         IVarEnd = Class->ivar_end();
           IVar != IVarEnd; ++IVar)
        Results.MaybeAddResult(Result(*IVar, 0), CurContext);
    }
  }
  
  // FIXME: How do we cope with isa?
  
  Results.ExitScope();

  // Add macros
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, NextRank, Results);

  // Hand off the results found for code completion.
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteTag(Scope *S, unsigned TagSpec) {
  if (!CodeCompleter)
    return;
  
  typedef CodeCompleteConsumer::Result Result;
  ResultBuilder::LookupFilter Filter = 0;
  switch ((DeclSpec::TST)TagSpec) {
  case DeclSpec::TST_enum:
    Filter = &ResultBuilder::IsEnum;
    break;
    
  case DeclSpec::TST_union:
    Filter = &ResultBuilder::IsUnion;
    break;
    
  case DeclSpec::TST_struct:
  case DeclSpec::TST_class:
    Filter = &ResultBuilder::IsClassOrStruct;
    break;
    
  default:
    assert(false && "Unknown type specifier kind in CodeCompleteTag");
    return;
  }
  
  ResultBuilder Results(*this, Filter);
  unsigned NextRank = CollectLookupResults(S, Context.getTranslationUnitDecl(), 
                                           0, CurContext, Results);
  
  if (getLangOptions().CPlusPlus) {
    // We could have the start of a nested-name-specifier. Add those
    // results as well.
    Results.setFilter(&ResultBuilder::IsNestedNameSpecifier);
    NextRank = CollectLookupResults(S, Context.getTranslationUnitDecl(), 
                                    NextRank, CurContext, Results);
  }
  
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, NextRank, Results);
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteCase(Scope *S) {
  if (getSwitchStack().empty() || !CodeCompleter)
    return;
  
  SwitchStmt *Switch = getSwitchStack().back();
  if (!Switch->getCond()->getType()->isEnumeralType())
    return;
  
  // Code-complete the cases of a switch statement over an enumeration type
  // by providing the list of 
  EnumDecl *Enum = Switch->getCond()->getType()->getAs<EnumType>()->getDecl();
  
  // Determine which enumerators we have already seen in the switch statement.
  // FIXME: Ideally, we would also be able to look *past* the code-completion
  // token, in case we are code-completing in the middle of the switch and not
  // at the end. However, we aren't able to do so at the moment.
  llvm::SmallPtrSet<EnumConstantDecl *, 8> EnumeratorsSeen;
  NestedNameSpecifier *Qualifier = 0;
  for (SwitchCase *SC = Switch->getSwitchCaseList(); SC; 
       SC = SC->getNextSwitchCase()) {
    CaseStmt *Case = dyn_cast<CaseStmt>(SC);
    if (!Case)
      continue;

    Expr *CaseVal = Case->getLHS()->IgnoreParenCasts();
    if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(CaseVal))
      if (EnumConstantDecl *Enumerator 
            = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
        // We look into the AST of the case statement to determine which 
        // enumerator was named. Alternatively, we could compute the value of 
        // the integral constant expression, then compare it against the
        // values of each enumerator. However, value-based approach would not 
        // work as well with C++ templates where enumerators declared within a 
        // template are type- and value-dependent.
        EnumeratorsSeen.insert(Enumerator);
        
        // If this is a qualified-id, keep track of the nested-name-specifier
        // so that we can reproduce it as part of code completion, e.g.,
        //
        //   switch (TagD.getKind()) {
        //     case TagDecl::TK_enum:
        //       break;
        //     case XXX
        //
        // At the XXX, our completions are TagDecl::TK_union,
        // TagDecl::TK_struct, and TagDecl::TK_class, rather than TK_union,
        // TK_struct, and TK_class.
        Qualifier = DRE->getQualifier();
      }
  }
  
  if (getLangOptions().CPlusPlus && !Qualifier && EnumeratorsSeen.empty()) {
    // If there are no prior enumerators in C++, check whether we have to 
    // qualify the names of the enumerators that we suggest, because they
    // may not be visible in this scope.
    Qualifier = getRequiredQualification(Context, CurContext,
                                         Enum->getDeclContext());
    
    // FIXME: Scoped enums need to start with "EnumDecl" as the context!
  }
  
  // Add any enumerators that have not yet been mentioned.
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  for (EnumDecl::enumerator_iterator E = Enum->enumerator_begin(),
                                  EEnd = Enum->enumerator_end();
       E != EEnd; ++E) {
    if (EnumeratorsSeen.count(*E))
      continue;
    
    Results.MaybeAddResult(CodeCompleteConsumer::Result(*E, 0, Qualifier));
  }
  Results.ExitScope();
  
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, 1, Results);
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

namespace {
  struct IsBetterOverloadCandidate {
    Sema &S;
    
  public:
    explicit IsBetterOverloadCandidate(Sema &S) : S(S) { }
    
    bool 
    operator()(const OverloadCandidate &X, const OverloadCandidate &Y) const {
      return S.isBetterOverloadCandidate(X, Y);
    }
  };
}

void Sema::CodeCompleteCall(Scope *S, ExprTy *FnIn,
                            ExprTy **ArgsIn, unsigned NumArgs) {
  if (!CodeCompleter)
    return;
  
  Expr *Fn = (Expr *)FnIn;
  Expr **Args = (Expr **)ArgsIn;
  
  // Ignore type-dependent call expressions entirely.
  if (Fn->isTypeDependent() || 
      Expr::hasAnyTypeDependentArguments(Args, NumArgs))
    return;
  
  NamedDecl *Function;
  DeclarationName UnqualifiedName;
  NestedNameSpecifier *Qualifier;
  SourceRange QualifierRange;
  bool ArgumentDependentLookup;
  bool HasExplicitTemplateArgs;
  const TemplateArgumentLoc *ExplicitTemplateArgs;
  unsigned NumExplicitTemplateArgs;
  
  DeconstructCallFunction(Fn,
                          Function, UnqualifiedName, Qualifier, QualifierRange,
                          ArgumentDependentLookup, HasExplicitTemplateArgs,
                          ExplicitTemplateArgs, NumExplicitTemplateArgs);

  
  // FIXME: What if we're calling something that isn't a function declaration?
  // FIXME: What if we're calling a pseudo-destructor?
  // FIXME: What if we're calling a member function?
  
  // Build an overload candidate set based on the functions we find.
  OverloadCandidateSet CandidateSet;
  AddOverloadedCallCandidates(Function, UnqualifiedName, 
                              ArgumentDependentLookup, HasExplicitTemplateArgs,
                              ExplicitTemplateArgs, NumExplicitTemplateArgs,
                              Args, NumArgs,
                              CandidateSet,
                              /*PartialOverloading=*/true);
  
  // Sort the overload candidate set by placing the best overloads first.
  std::stable_sort(CandidateSet.begin(), CandidateSet.end(),
                   IsBetterOverloadCandidate(*this));
  
  // Add the remaining viable overload candidates as code-completion reslults.  
  typedef CodeCompleteConsumer::OverloadCandidate ResultCandidate;
  llvm::SmallVector<ResultCandidate, 8> Results;
  
  for (OverloadCandidateSet::iterator Cand = CandidateSet.begin(),
                                   CandEnd = CandidateSet.end();
       Cand != CandEnd; ++Cand) {
    if (Cand->Viable)
      Results.push_back(ResultCandidate(Cand->Function));
  }
  CodeCompleter->ProcessOverloadCandidates(*this, NumArgs, Results.data(), 
                                           Results.size());
}

void Sema::CodeCompleteQualifiedId(Scope *S, const CXXScopeSpec &SS,
                                   bool EnteringContext) {
  if (!SS.getScopeRep() || !CodeCompleter)
    return;
  
  DeclContext *Ctx = computeDeclContext(SS, EnteringContext);
  if (!Ctx)
    return;
  
  ResultBuilder Results(*this);
  unsigned NextRank = CollectMemberLookupResults(Ctx, 0, Ctx, Results);
  
  // The "template" keyword can follow "::" in the grammar, but only
  // put it into the grammar if the nested-name-specifier is dependent.
  NestedNameSpecifier *NNS = (NestedNameSpecifier *)SS.getScopeRep();
  if (!Results.empty() && NNS->isDependent())
    Results.MaybeAddResult(CodeCompleteConsumer::Result("template", NextRank));
  
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, NextRank + 1, Results);
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteUsing(Scope *S) {
  if (!CodeCompleter)
    return;
  
  ResultBuilder Results(*this, &ResultBuilder::IsNestedNameSpecifier);
  Results.EnterNewScope();
  
  // If we aren't in class scope, we could see the "namespace" keyword.
  if (!S->isClassScope())
    Results.MaybeAddResult(CodeCompleteConsumer::Result("namespace", 0));
  
  // After "using", we can see anything that would start a 
  // nested-name-specifier.
  unsigned NextRank = CollectLookupResults(S, Context.getTranslationUnitDecl(), 
                                           0, CurContext, Results);
  Results.ExitScope();
  
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, NextRank, Results);
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteUsingDirective(Scope *S) {
  if (!CodeCompleter)
    return;
  
  // After "using namespace", we expect to see a namespace name or namespace
  // alias.
  ResultBuilder Results(*this, &ResultBuilder::IsNamespaceOrAlias);
  Results.EnterNewScope();
  unsigned NextRank = CollectLookupResults(S, Context.getTranslationUnitDecl(), 
                                           0, CurContext, Results);
  Results.ExitScope();
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, NextRank, Results);
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteNamespaceDecl(Scope *S)  {
  if (!CodeCompleter)
    return;
  
  ResultBuilder Results(*this, &ResultBuilder::IsNamespace);
  DeclContext *Ctx = (DeclContext *)S->getEntity();
  if (!S->getParent())
    Ctx = Context.getTranslationUnitDecl();
  
  if (Ctx && Ctx->isFileContext()) {
    // We only want to see those namespaces that have already been defined
    // within this scope, because its likely that the user is creating an
    // extended namespace declaration. Keep track of the most recent 
    // definition of each namespace.
    std::map<NamespaceDecl *, NamespaceDecl *> OrigToLatest;
    for (DeclContext::specific_decl_iterator<NamespaceDecl> 
         NS(Ctx->decls_begin()), NSEnd(Ctx->decls_end());
         NS != NSEnd; ++NS)
      OrigToLatest[NS->getOriginalNamespace()] = *NS;
    
    // Add the most recent definition (or extended definition) of each 
    // namespace to the list of results.
    Results.EnterNewScope();
    for (std::map<NamespaceDecl *, NamespaceDecl *>::iterator 
         NS = OrigToLatest.begin(), NSEnd = OrigToLatest.end();
         NS != NSEnd; ++NS)
      Results.MaybeAddResult(CodeCompleteConsumer::Result(NS->second, 0),
                             CurContext);
    Results.ExitScope();
  }
  
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, 1, Results);
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteNamespaceAliasDecl(Scope *S)  {
  if (!CodeCompleter)
    return;
  
  // After "namespace", we expect to see a namespace or alias.
  ResultBuilder Results(*this, &ResultBuilder::IsNamespaceOrAlias);
  unsigned NextRank = CollectLookupResults(S, Context.getTranslationUnitDecl(), 
                                           0, CurContext, Results);
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, NextRank, Results);
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteOperatorName(Scope *S) {
  if (!CodeCompleter)
    return;

  typedef CodeCompleteConsumer::Result Result;
  ResultBuilder Results(*this, &ResultBuilder::IsType);
  Results.EnterNewScope();
  
  // Add the names of overloadable operators.
#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly)      \
  if (std::strcmp(Spelling, "?"))                                                  \
    Results.MaybeAddResult(Result(Spelling, 0));
#include "clang/Basic/OperatorKinds.def"
  
  // Add any type names visible from the current scope
  unsigned NextRank = CollectLookupResults(S, Context.getTranslationUnitDecl(), 
                                           0, CurContext, Results);
  
  // Add any type specifiers
  AddTypeSpecifierResults(getLangOptions(), 0, Results);
  
  // Add any nested-name-specifiers
  Results.setFilter(&ResultBuilder::IsNestedNameSpecifier);
  NextRank = CollectLookupResults(S, Context.getTranslationUnitDecl(), 
                                  NextRank + 1, CurContext, Results);
  Results.ExitScope();
  
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, NextRank, Results);
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteObjCProperty(Scope *S, ObjCDeclSpec &ODS) { 
  if (!CodeCompleter)
    return;
  unsigned Attributes = ODS.getPropertyAttributes();
  
  typedef CodeCompleteConsumer::Result Result;
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  if (!(Attributes & ObjCDeclSpec::DQ_PR_readonly))
    Results.MaybeAddResult(CodeCompleteConsumer::Result("readonly", 0));
  if (!(Attributes & ObjCDeclSpec::DQ_PR_assign))
    Results.MaybeAddResult(CodeCompleteConsumer::Result("assign", 0));
  if (!(Attributes & ObjCDeclSpec::DQ_PR_readwrite))
    Results.MaybeAddResult(CodeCompleteConsumer::Result("readwrite", 0));
  if (!(Attributes & ObjCDeclSpec::DQ_PR_retain))
    Results.MaybeAddResult(CodeCompleteConsumer::Result("retain", 0));
  if (!(Attributes & ObjCDeclSpec::DQ_PR_copy))
    Results.MaybeAddResult(CodeCompleteConsumer::Result("copy", 0));
  if (!(Attributes & ObjCDeclSpec::DQ_PR_nonatomic))
    Results.MaybeAddResult(CodeCompleteConsumer::Result("nonatomic", 0));
  if (!(Attributes & ObjCDeclSpec::DQ_PR_setter))
    Results.MaybeAddResult(CodeCompleteConsumer::Result("setter", 0));
  if (!(Attributes & ObjCDeclSpec::DQ_PR_getter))
    Results.MaybeAddResult(CodeCompleteConsumer::Result("getter", 0));
  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

/// \brief Add all of the Objective-C methods in the given Objective-C 
/// container to the set of results.
///
/// The container will be a class, protocol, category, or implementation of 
/// any of the above. This mether will recurse to include methods from 
/// the superclasses of classes along with their categories, protocols, and
/// implementations.
///
/// \param Container the container in which we'll look to find methods.
///
/// \param WantInstance whether to add instance methods (only); if false, this
/// routine will add factory methods (only).
///
/// \param CurContext the context in which we're performing the lookup that
/// finds methods.
///
/// \param Results the structure into which we'll add results.
static void AddObjCMethods(ObjCContainerDecl *Container, 
                           bool WantInstanceMethods,
                           DeclContext *CurContext,
                           ResultBuilder &Results) {
  typedef CodeCompleteConsumer::Result Result;
  for (ObjCContainerDecl::method_iterator M = Container->meth_begin(),
                                       MEnd = Container->meth_end();
       M != MEnd; ++M) {
    if ((*M)->isInstanceMethod() == WantInstanceMethods)
      Results.MaybeAddResult(Result(*M, 0), CurContext);
  }
  
  ObjCInterfaceDecl *IFace = dyn_cast<ObjCInterfaceDecl>(Container);
  if (!IFace)
    return;
  
  // Add methods in protocols.
  const ObjCList<ObjCProtocolDecl> &Protocols= IFace->getReferencedProtocols();
  for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
                                            E = Protocols.end(); 
       I != E; ++I)
    AddObjCMethods(*I, WantInstanceMethods, CurContext, Results);
  
  // Add methods in categories.
  for (ObjCCategoryDecl *CatDecl = IFace->getCategoryList(); CatDecl;
       CatDecl = CatDecl->getNextClassCategory()) {
    AddObjCMethods(CatDecl, WantInstanceMethods, CurContext, Results);
    
    // Add a categories protocol methods.
    const ObjCList<ObjCProtocolDecl> &Protocols 
      = CatDecl->getReferencedProtocols();
    for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
                                              E = Protocols.end();
         I != E; ++I)
      AddObjCMethods(*I, WantInstanceMethods, CurContext, Results);
    
    // Add methods in category implementations.
    if (ObjCCategoryImplDecl *Impl = CatDecl->getImplementation())
      AddObjCMethods(Impl, WantInstanceMethods, CurContext, Results);
  }
  
  // Add methods in superclass.
  if (IFace->getSuperClass())
    AddObjCMethods(IFace->getSuperClass(), WantInstanceMethods, CurContext,
                   Results);

  // Add methods in our implementation, if any.
  if (ObjCImplementationDecl *Impl = IFace->getImplementation())
    AddObjCMethods(Impl, WantInstanceMethods, CurContext, Results);
}

void Sema::CodeCompleteObjCClassMessage(Scope *S, IdentifierInfo *FName,
                                        SourceLocation FNameLoc) {
  typedef CodeCompleteConsumer::Result Result;
  ObjCInterfaceDecl *CDecl = 0;

  if (FName->isStr("super")) {
    // We're sending a message to "super".
    if (ObjCMethodDecl *CurMethod = getCurMethodDecl()) {
      // Figure out which interface we're in.
      CDecl = CurMethod->getClassInterface();
      if (!CDecl)
        return;

      // Find the superclass of this class.
      CDecl = CDecl->getSuperClass();
      if (!CDecl)
        return;

      if (CurMethod->isInstanceMethod()) {
        // We are inside an instance method, which means that the message
        // send [super ...] is actually calling an instance method on the
        // current object. Build the super expression and handle this like
        // an instance method.
        QualType SuperTy = Context.getObjCInterfaceType(CDecl);
        SuperTy = Context.getObjCObjectPointerType(SuperTy);
        OwningExprResult Super
          = Owned(new (Context) ObjCSuperExpr(FNameLoc, SuperTy));
        return CodeCompleteObjCInstanceMessage(S, (Expr *)Super.get());
      }

      // Okay, we're calling a factory method in our superclass.
    } 
  }

  // If the given name refers to an interface type, retrieve the
  // corresponding declaration.
  if (!CDecl)
    if (TypeTy *Ty = getTypeName(*FName, FNameLoc, S, 0, false)) {
      QualType T = GetTypeFromParser(Ty, 0);
      if (!T.isNull()) 
        if (const ObjCInterfaceType *Interface = T->getAs<ObjCInterfaceType>())
          CDecl = Interface->getDecl();
    }

  if (!CDecl && FName->isStr("super")) {
    // "super" may be the name of a variable, in which case we are
    // probably calling an instance method.
    OwningExprResult Super = ActOnDeclarationNameExpr(S, FNameLoc, FName,
                                                      false, 0, false);
    return CodeCompleteObjCInstanceMessage(S, (Expr *)Super.get());
  }

  // Add all of the factory methods in this Objective-C class, its protocols,
  // superclasses, categories, implementation, etc.
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  AddObjCMethods(CDecl, false, CurContext, Results);  
  Results.ExitScope();
  
  // This also suppresses remaining diagnostics.
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteObjCInstanceMessage(Scope *S, ExprTy *Receiver) {
  typedef CodeCompleteConsumer::Result Result;
  
  Expr *RecExpr = static_cast<Expr *>(Receiver);
  QualType RecType = RecExpr->getType();
  
  // If necessary, apply function/array conversion to the receiver.
  // C99 6.7.5.3p[7,8].
  DefaultFunctionArrayConversion(RecExpr);
  QualType ReceiverType = RecExpr->getType();
  
  if (ReceiverType->isObjCIdType() || ReceiverType->isBlockPointerType()) {
    // FIXME: We're messaging 'id'. Do we actually want to look up every method
    // in the universe?
    return;
  }
  
  // Build the set of methods we can see.
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  
  // Handle messages to Class. This really isn't a message to an instance
  // method, so we treat it the same way we would treat a message send to a
  // class method.
  if (ReceiverType->isObjCClassType() || 
      ReceiverType->isObjCQualifiedClassType()) {
    if (ObjCMethodDecl *CurMethod = getCurMethodDecl()) {
      if (ObjCInterfaceDecl *ClassDecl = CurMethod->getClassInterface())
        AddObjCMethods(ClassDecl, false, CurContext, Results);
    }
  } 
  // Handle messages to a qualified ID ("id<foo>").
  else if (const ObjCObjectPointerType *QualID
             = ReceiverType->getAsObjCQualifiedIdType()) {
    // Search protocols for instance methods.
    for (ObjCObjectPointerType::qual_iterator I = QualID->qual_begin(),
                                              E = QualID->qual_end(); 
         I != E; ++I)
      AddObjCMethods(*I, true, CurContext, Results);
  }
  // Handle messages to a pointer to interface type.
  else if (const ObjCObjectPointerType *IFacePtr
                              = ReceiverType->getAsObjCInterfacePointerType()) {
    // Search the class, its superclasses, etc., for instance methods.
    AddObjCMethods(IFacePtr->getInterfaceDecl(), true, CurContext, Results);
    
    // Search protocols for instance methods.
    for (ObjCObjectPointerType::qual_iterator I = IFacePtr->qual_begin(),
         E = IFacePtr->qual_end(); 
         I != E; ++I)
      AddObjCMethods(*I, true, CurContext, Results);
  }
  
  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

/// \brief Add all of the protocol declarations that we find in the given
/// (translation unit) context.
static void AddProtocolResults(DeclContext *Ctx, DeclContext *CurContext,
                               bool OnlyForwardDeclarations,
                               ResultBuilder &Results) {
  typedef CodeCompleteConsumer::Result Result;
  
  for (DeclContext::decl_iterator D = Ctx->decls_begin(), 
                               DEnd = Ctx->decls_end();
       D != DEnd; ++D) {
    // Record any protocols we find.
    if (ObjCProtocolDecl *Proto = dyn_cast<ObjCProtocolDecl>(*D))
      if (!OnlyForwardDeclarations || Proto->isForwardDecl())
        Results.MaybeAddResult(Result(Proto, 0), CurContext);

    // Record any forward-declared protocols we find.
    if (ObjCForwardProtocolDecl *Forward
          = dyn_cast<ObjCForwardProtocolDecl>(*D)) {
      for (ObjCForwardProtocolDecl::protocol_iterator 
             P = Forward->protocol_begin(),
             PEnd = Forward->protocol_end();
           P != PEnd; ++P)
        if (!OnlyForwardDeclarations || (*P)->isForwardDecl())
          Results.MaybeAddResult(Result(*P, 0), CurContext);
    }
  }
}

void Sema::CodeCompleteObjCProtocolReferences(IdentifierLocPair *Protocols,
                                              unsigned NumProtocols) {
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  
  // Tell the result set to ignore all of the protocols we have
  // already seen.
  for (unsigned I = 0; I != NumProtocols; ++I)
    if (ObjCProtocolDecl *Protocol = LookupProtocol(Protocols[I].first))
      Results.Ignore(Protocol);

  // Add all protocols.
  AddProtocolResults(Context.getTranslationUnitDecl(), CurContext, false,
                     Results);

  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteObjCProtocolDecl(Scope *) {
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  
  // Add all protocols.
  AddProtocolResults(Context.getTranslationUnitDecl(), CurContext, true,
                     Results);

  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}
