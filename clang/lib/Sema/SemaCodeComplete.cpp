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
#include "llvm/ADT/SmallPtrSet.h"
#include <list>
#include <map>
#include <vector>

using namespace clang;

/// \brief Set the code-completion consumer for semantic analysis.
void Sema::setCodeCompleteConsumer(CodeCompleteConsumer *CCC) {
  assert(((CodeCompleter != 0) != (CCC != 0)) && 
         "Already set or cleared a code-completion consumer?");
  CodeCompleter = CCC;
}

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
  
  // Look through using declarations.
  if (UsingDecl *Using = dyn_cast<UsingDecl>(R.Declaration))
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
    
    // FIXME: Should we filter out other names in the implementation's
    // namespace, e.g., those containing a __ or that start with _[A-Z]?
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
  
  // If this result is supposed to have an informative qualifier, add one.
  if (R.QualifierIsInformative && !R.Qualifier) {
    DeclContext *Ctx = R.Declaration->getDeclContext();
    if (NamespaceDecl *Namespace = dyn_cast<NamespaceDecl>(Ctx))
      R.Qualifier = NestedNameSpecifier::Create(SemaRef.Context, 0, Namespace);
    else if (TagDecl *Tag = dyn_cast<TagDecl>(Ctx))
      R.Qualifier = NestedNameSpecifier::Create(SemaRef.Context, 0, false, 
                             SemaRef.Context.getTypeDeclType(Tag).getTypePtr());
    else
      R.QualifierIsInformative = false;
  }
  
  // If the filter is for nested-name-specifiers, then this result starts a
  // nested-name-specifier.
  if (Filter == &ResultBuilder::IsNestedNameSpecifier)
    R.StartsNestedNameSpecifier = true;
  
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
      CCStr->AddTextChunk(", ");
    
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
      CCStr->AddTextChunk(", ");
    
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
  if (Kind != RK_Declaration)
    return 0;
  
  NamedDecl *ND = Declaration;
  
  if (FunctionDecl *Function = dyn_cast<FunctionDecl>(ND)) {
    CodeCompletionString *Result = new CodeCompletionString;
    AddQualifierToCompletionString(Result, Qualifier, QualifierIsInformative, 
                                   S.Context);
    Result->AddTextChunk(Function->getNameAsString().c_str());
    Result->AddTextChunk("(");
    AddFunctionParameterChunks(S.Context, Function, Result);
    Result->AddTextChunk(")");
    return Result;
  }
  
  if (FunctionTemplateDecl *FunTmpl = dyn_cast<FunctionTemplateDecl>(ND)) {
    CodeCompletionString *Result = new CodeCompletionString;
    AddQualifierToCompletionString(Result, Qualifier, QualifierIsInformative, 
                                   S.Context);
    FunctionDecl *Function = FunTmpl->getTemplatedDecl();
    Result->AddTextChunk(Function->getNameAsString().c_str());
    
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
      Result->AddTextChunk("<");
      AddTemplateParameterChunks(S.Context, FunTmpl, Result, 
                                 LastDeducibleArgument);
      Result->AddTextChunk(">");
    }
    
    // Add the function parameters
    Result->AddTextChunk("(");
    AddFunctionParameterChunks(S.Context, Function, Result);
    Result->AddTextChunk(")");
    return Result;
  }
  
  if (TemplateDecl *Template = dyn_cast<TemplateDecl>(ND)) {
    CodeCompletionString *Result = new CodeCompletionString;
    AddQualifierToCompletionString(Result, Qualifier, QualifierIsInformative, 
                                   S.Context);
    Result->AddTextChunk(Template->getNameAsString().c_str());
    Result->AddTextChunk("<");
    AddTemplateParameterChunks(S.Context, Template, Result);
    Result->AddTextChunk(">");
    return Result;
  }
  
  if (Qualifier || StartsNestedNameSpecifier) {
    CodeCompletionString *Result = new CodeCompletionString;
    AddQualifierToCompletionString(Result, Qualifier, QualifierIsInformative, 
                                   S.Context);
    Result->AddTextChunk(ND->getNameAsString().c_str());
    if (StartsNestedNameSpecifier)
      Result->AddTextChunk("::");
    return Result;
  }
  
  return 0;
}

CodeCompletionString *
CodeCompleteConsumer::OverloadCandidate::CreateSignatureString(
                                                          unsigned CurrentArg,
                                                               Sema &S) const {
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
    Result->AddTextChunk("(");
    Result->AddPlaceholderChunk("...");
    Result->AddTextChunk("(");    
    return Result;
  }
  
  if (FDecl)
    Result->AddTextChunk(FDecl->getNameAsString().c_str());    
  else
    Result->AddTextChunk(
         Proto->getResultType().getAsString(S.Context.PrintingPolicy).c_str());
  
  Result->AddTextChunk("(");
  unsigned NumParams = FDecl? FDecl->getNumParams() : Proto->getNumArgs();
  for (unsigned I = 0; I != NumParams; ++I) {
    if (I)
      Result->AddTextChunk(", ");
    
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
      Result->AddPlaceholderChunk(ArgString.c_str());
    else
      Result->AddTextChunk(ArgString.c_str());
  }
  
  if (Proto && Proto->isVariadic()) {
    Result->AddTextChunk(", ");
    if (CurrentArg < NumParams)
      Result->AddTextChunk("...");
    else
      Result->AddPlaceholderChunk("...");
  }
  Result->AddTextChunk(")");
  
  return Result;
}

namespace {
  struct SortCodeCompleteResult {
    typedef CodeCompleteConsumer::Result Result;
    
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
      
      // Ordering depends on the kind of result.
      switch (X.Kind) {
        case Result::RK_Declaration:
          // Order based on the declaration names.
          return X.Declaration->getDeclName() < Y.Declaration->getDeclName();
          
        case Result::RK_Keyword:
          return strcmp(X.Keyword, Y.Keyword) == -1;
      }
      
      // Silence GCC warning.
      return false;
    }
  };
}

static void HandleCodeCompleteResults(CodeCompleteConsumer *CodeCompleter,
                                      CodeCompleteConsumer::Result *Results,
                                      unsigned NumResults) {
  // Sort the results by rank/kind/etc.
  std::stable_sort(Results, Results + NumResults, SortCodeCompleteResult());

  if (CodeCompleter)
    CodeCompleter->ProcessCodeCompleteResults(Results, NumResults);
}

void Sema::CodeCompleteOrdinaryName(Scope *S) {
  ResultBuilder Results(*this, &ResultBuilder::IsOrdinaryName);
  CollectLookupResults(S, Context.getTranslationUnitDecl(), 0, CurContext, 
                       Results);
  HandleCodeCompleteResults(CodeCompleter, Results.data(), Results.size());
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
  
  ResultBuilder Results(*this);
  unsigned NextRank = 0;
  
  if (const RecordType *Record = BaseType->getAs<RecordType>()) {
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
    
    // Hand off the results found for code completion.
    HandleCodeCompleteResults(CodeCompleter, Results.data(), Results.size());
    
    // We're done!
    return;
  }
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
    CollectLookupResults(S, Context.getTranslationUnitDecl(), NextRank, 
                         CurContext, Results);
  }
  
  HandleCodeCompleteResults(CodeCompleter, Results.data(), Results.size());
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
        if (QualifiedDeclRefExpr *QDRE = dyn_cast<QualifiedDeclRefExpr>(DRE))
          Qualifier = QDRE->getQualifier();
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
  
  HandleCodeCompleteResults(CodeCompleter, Results.data(), Results.size());  
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
  const TemplateArgument *ExplicitTemplateArgs;
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
  CodeCompleter->ProcessOverloadCandidates(NumArgs, Results.data(), 
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
  
  HandleCodeCompleteResults(CodeCompleter, Results.data(), Results.size());
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
  CollectLookupResults(S, Context.getTranslationUnitDecl(), 0, 
                       CurContext, Results);
  Results.ExitScope();
  
  HandleCodeCompleteResults(CodeCompleter, Results.data(), Results.size());
}

void Sema::CodeCompleteUsingDirective(Scope *S) {
  if (!CodeCompleter)
    return;
  
  // After "using namespace", we expect to see a namespace name or namespace
  // alias.
  ResultBuilder Results(*this, &ResultBuilder::IsNamespaceOrAlias);
  Results.EnterNewScope();
  CollectLookupResults(S, Context.getTranslationUnitDecl(), 0, CurContext,
                       Results);
  Results.ExitScope();
  HandleCodeCompleteResults(CodeCompleter, Results.data(), Results.size());  
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
  
  HandleCodeCompleteResults(CodeCompleter, Results.data(), Results.size());  
}

void Sema::CodeCompleteNamespaceAliasDecl(Scope *S)  {
  if (!CodeCompleter)
    return;
  
  // After "namespace", we expect to see a namespace or alias.
  ResultBuilder Results(*this, &ResultBuilder::IsNamespaceOrAlias);
  CollectLookupResults(S, Context.getTranslationUnitDecl(), 0, CurContext,
                       Results);
  HandleCodeCompleteResults(CodeCompleter, Results.data(), Results.size());  
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
  CollectLookupResults(S, Context.getTranslationUnitDecl(), NextRank + 1, 
                       CurContext, Results);
  Results.ExitScope();
  
  HandleCodeCompleteResults(CodeCompleter, Results.data(), Results.size());  
}

