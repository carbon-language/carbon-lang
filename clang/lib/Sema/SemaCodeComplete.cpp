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
    
    typedef std::pair<NamedDecl *, unsigned> DeclIndexPair;

    /// \brief An entry in the shadow map, which is optimized to store
    /// a single (declaration, index) mapping (the common case) but
    /// can also store a list of (declaration, index) mappings.
    class ShadowMapEntry {
      typedef llvm::SmallVector<DeclIndexPair, 4> DeclIndexPairVector;

      /// \brief Contains either the solitary NamedDecl * or a vector
      /// of (declaration, index) pairs.
      llvm::PointerUnion<NamedDecl *, DeclIndexPairVector*> DeclOrVector;

      /// \brief When the entry contains a single declaration, this is
      /// the index associated with that entry.
      unsigned SingleDeclIndex;

    public:
      ShadowMapEntry() : DeclOrVector(), SingleDeclIndex(0) { }

      void Add(NamedDecl *ND, unsigned Index) {
        if (DeclOrVector.isNull()) {
          // 0 - > 1 elements: just set the single element information.
          DeclOrVector = ND;
          SingleDeclIndex = Index;
          return;
        }

        if (NamedDecl *PrevND = DeclOrVector.dyn_cast<NamedDecl *>()) {
          // 1 -> 2 elements: create the vector of results and push in the
          // existing declaration.
          DeclIndexPairVector *Vec = new DeclIndexPairVector;
          Vec->push_back(DeclIndexPair(PrevND, SingleDeclIndex));
          DeclOrVector = Vec;
        }

        // Add the new element to the end of the vector.
        DeclOrVector.get<DeclIndexPairVector*>()->push_back(
                                                    DeclIndexPair(ND, Index));
      }

      void Destroy() {
        if (DeclIndexPairVector *Vec
              = DeclOrVector.dyn_cast<DeclIndexPairVector *>()) {
          delete Vec;
          DeclOrVector = ((NamedDecl *)0);
        }
      }

      // Iteration.
      class iterator;
      iterator begin() const;
      iterator end() const;
    };

    /// \brief A mapping from declaration names to the declarations that have
    /// this name within a particular scope and their index within the list of
    /// results.
    typedef llvm::DenseMap<DeclarationName, ShadowMapEntry> ShadowMap;
    
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
    bool IsOrdinaryNonValueName(NamedDecl *ND) const;
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

class ResultBuilder::ShadowMapEntry::iterator {
  llvm::PointerUnion<NamedDecl*, const DeclIndexPair*> DeclOrIterator;
  unsigned SingleDeclIndex;

public:
  typedef DeclIndexPair value_type;
  typedef value_type reference;
  typedef std::ptrdiff_t difference_type;
  typedef std::input_iterator_tag iterator_category;
        
  class pointer {
    DeclIndexPair Value;

  public:
    pointer(const DeclIndexPair &Value) : Value(Value) { }

    const DeclIndexPair *operator->() const {
      return &Value;
    }
  };
        
  iterator() : DeclOrIterator((NamedDecl *)0), SingleDeclIndex(0) { }

  iterator(NamedDecl *SingleDecl, unsigned Index)
    : DeclOrIterator(SingleDecl), SingleDeclIndex(Index) { }

  iterator(const DeclIndexPair *Iterator)
    : DeclOrIterator(Iterator), SingleDeclIndex(0) { }

  iterator &operator++() {
    if (DeclOrIterator.is<NamedDecl *>()) {
      DeclOrIterator = (NamedDecl *)0;
      SingleDeclIndex = 0;
      return *this;
    }

    const DeclIndexPair *I = DeclOrIterator.get<const DeclIndexPair*>();
    ++I;
    DeclOrIterator = I;
    return *this;
  }

  iterator operator++(int) {
    iterator tmp(*this);
    ++(*this);
    return tmp;
  }

  reference operator*() const {
    if (NamedDecl *ND = DeclOrIterator.dyn_cast<NamedDecl *>())
      return reference(ND, SingleDeclIndex);

    return *DeclOrIterator.get<const DeclIndexPair*>();
  }

  pointer operator->() const {
    return pointer(**this);
  }

  friend bool operator==(const iterator &X, const iterator &Y) {
    return X.DeclOrIterator.getOpaqueValue()
                                  == Y.DeclOrIterator.getOpaqueValue() &&
      X.SingleDeclIndex == Y.SingleDeclIndex;
  }

  friend bool operator!=(const iterator &X, const iterator &Y) {
    return !(X == Y);
  }
};

ResultBuilder::ShadowMapEntry::iterator 
ResultBuilder::ShadowMapEntry::begin() const {
  if (DeclOrVector.isNull())
    return iterator();

  if (NamedDecl *ND = DeclOrVector.dyn_cast<NamedDecl *>())
    return iterator(ND, SingleDeclIndex);

  return iterator(DeclOrVector.get<DeclIndexPairVector *>()->begin());
}

ResultBuilder::ShadowMapEntry::iterator 
ResultBuilder::ShadowMapEntry::end() const {
  if (DeclOrVector.is<NamedDecl *>() || DeclOrVector.isNull())
    return iterator();

  return iterator(DeclOrVector.get<DeclIndexPairVector *>()->end());
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
    MaybeAddResult(Result(Using->getTargetDecl(), R.Qualifier), CurContext);
  
  Decl *CanonDecl = R.Declaration->getCanonicalDecl();
  unsigned IDNS = CanonDecl->getIdentifierNamespace();
  
  // Friend declarations and declarations introduced due to friends are never
  // added as results.
  if (isa<FriendDecl>(CanonDecl) || 
      (IDNS & (Decl::IDNS_OrdinaryFriend | Decl::IDNS_TagFriend)))
    return;

  // Class template (partial) specializations are never added as results.
  if (isa<ClassTemplateSpecializationDecl>(CanonDecl) ||
      isa<ClassTemplatePartialSpecializationDecl>(CanonDecl))
    return;
  
  // Using declarations themselves are never added as results.
  if (isa<UsingDecl>(CanonDecl))
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
  ShadowMapEntry::iterator I, IEnd;
  ShadowMap::iterator NamePos = SMap.find(R.Declaration->getDeclName());
  if (NamePos != SMap.end()) {
    I = NamePos->second.begin();
    IEnd = NamePos->second.end();
  }

  for (; I != IEnd; ++I) {
    NamedDecl *ND = I->first;
    unsigned Index = I->second;
    if (ND->getCanonicalDecl() == CanonDecl) {
      // This is a redeclaration. Always pick the newer declaration.
      Results[Index].Declaration = R.Declaration;
      
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
    ShadowMapEntry::iterator I, IEnd;
    ShadowMap::iterator NamePos = SM->find(R.Declaration->getDeclName());
    if (NamePos != SM->end()) {
      I = NamePos->second.begin();
      IEnd = NamePos->second.end();
    }
    for (; I != IEnd; ++I) {
      // A tag declaration does not hide a non-tag declaration.
      if (I->first->getIdentifierNamespace() == Decl::IDNS_Tag &&
          (IDNS & (Decl::IDNS_Member | Decl::IDNS_Ordinary | 
                   Decl::IDNS_ObjCProtocol)))
        continue;
      
      // Protocols are in distinct namespaces from everything else.
      if (((I->first->getIdentifierNamespace() & Decl::IDNS_ObjCProtocol)
           || (IDNS & Decl::IDNS_ObjCProtocol)) &&
          I->first->getIdentifierNamespace() != IDNS)
        continue;
      
      // The newly-added result is hidden by an entry in the shadow map.
      if (canHiddenResultBeFound(SemaRef.getLangOptions(), R.Declaration, 
                                 I->first)) {
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
  SMap[R.Declaration->getDeclName()].Add(R.Declaration, Results.size());
  Results.push_back(R);
}

/// \brief Enter into a new scope.
void ResultBuilder::EnterNewScope() {
  ShadowMaps.push_back(ShadowMap());
}

/// \brief Exit from the current scope.
void ResultBuilder::ExitScope() {
  for (ShadowMap::iterator E = ShadowMaps.back().begin(),
                        EEnd = ShadowMaps.back().end();
       E != EEnd;
       ++E)
    E->second.Destroy();
         
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

/// \brief Determines whether this given declaration will be found by
/// ordinary name lookup.
bool ResultBuilder::IsOrdinaryNonValueName(NamedDecl *ND) const {
  unsigned IDNS = Decl::IDNS_Ordinary;
  if (SemaRef.getLangOptions().CPlusPlus)
    IDNS |= Decl::IDNS_Tag;
  
  return (ND->getIdentifierNamespace() & IDNS) && 
    !isa<ValueDecl>(ND) && !isa<FunctionTemplateDecl>(ND);
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

/// \brief Determines whether the given declaration is a type.
bool ResultBuilder::IsType(NamedDecl *ND) const {
  return isa<TypeDecl>(ND);
}

/// \brief Determines which members of a class should be visible via
/// "." or "->".  Only value declarations, nested name specifiers, and
/// using declarations thereof should show up.
bool ResultBuilder::IsMember(NamedDecl *ND) const {
  if (UsingShadowDecl *Using = dyn_cast<UsingShadowDecl>(ND))
    ND = Using->getTargetDecl();

  return isa<ValueDecl>(ND) || isa<FunctionTemplateDecl>(ND) ||
    isa<ObjCPropertyDecl>(ND);
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
/// \param Visited the set of declaration contexts that have already been
/// visited. Declaration contexts will only be visited once.
///
/// \param Results the result set that will be extended with any results
/// found within this declaration context (and, for a C++ class, its bases).
///
/// \param InBaseClass whether we are in a base class.
static void CollectMemberLookupResults(DeclContext *Ctx, 
                                       DeclContext *CurContext,
                                 llvm::SmallPtrSet<DeclContext *, 16> &Visited,
                                       ResultBuilder &Results,
                                       bool InBaseClass = false) {
  // Make sure we don't visit the same context twice.
  if (!Visited.insert(Ctx->getPrimaryContext()))
    return;
  
  // Enumerate all of the results in this context.
  typedef CodeCompleteConsumer::Result Result;
  Results.EnterNewScope();
  for (DeclContext *CurCtx = Ctx->getPrimaryContext(); CurCtx; 
       CurCtx = CurCtx->getNextContext()) {
    for (DeclContext::decl_iterator D = CurCtx->decls_begin(), 
                                 DEnd = CurCtx->decls_end();
         D != DEnd; ++D) {
      if (NamedDecl *ND = dyn_cast<NamedDecl>(*D))
        Results.MaybeAddResult(Result(ND, 0, InBaseClass), CurContext);
      
      // Visit transparent contexts inside this context.
      if (DeclContext *InnerCtx = dyn_cast<DeclContext>(*D)) {
        if (InnerCtx->isTransparentContext())
          CollectMemberLookupResults(InnerCtx, CurContext, Visited,
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
      CollectMemberLookupResults(Record->getDecl(), CurContext, Visited,
                                 Results, /*InBaseClass=*/true);
    }
  }
  
  // FIXME: Look into base classes in Objective-C!
  
  Results.ExitScope();
}

/// \brief Collect the results of searching for members within the given
/// declaration context.
///
/// \param Ctx the declaration context from which we will gather results.
///
/// \param Results the result set that will be extended with any results
/// found within this declaration context (and, for a C++ class, its bases).
static void CollectMemberLookupResults(DeclContext *Ctx, 
                                       DeclContext *CurContext,
                                       ResultBuilder &Results) {
  llvm::SmallPtrSet<DeclContext *, 16> Visited;
  CollectMemberLookupResults(Ctx, CurContext, Visited, Results);
}

/// \brief Collect the results of searching for declarations within the given
/// scope and its parent scopes.
///
/// \param S the scope in which we will start looking for declarations.
///
/// \param CurContext the context from which lookup results will be found.
///
/// \param Results the builder object that will receive each result.
static void CollectLookupResults(Scope *S, 
                                 TranslationUnitDecl *TranslationUnit,
                                 DeclContext *CurContext,
                                 ResultBuilder &Results) {
  if (!S)
    return;
  
  // FIXME: Using directives!
  
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
      
      CollectMemberLookupResults(Ctx, CurContext, Results);
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
    CollectMemberLookupResults(TranslationUnit, CurContext, Results);
  } else {
    // Walk through the declarations in this Scope.
    for (Scope::decl_iterator D = S->decl_begin(), DEnd = S->decl_end();
         D != DEnd; ++D) {
      if (NamedDecl *ND = dyn_cast<NamedDecl>((Decl *)((*D).get())))
        Results.MaybeAddResult(CodeCompleteConsumer::Result(ND), CurContext);        
    }
  }
  
  // Lookup names in the parent scope.
  CollectLookupResults(S->getParent(), TranslationUnit, CurContext, Results);
  Results.ExitScope();
}

/// \brief Add type specifiers for the current language as keyword results.
static void AddTypeSpecifierResults(const LangOptions &LangOpts,
                                    ResultBuilder &Results) {
  typedef CodeCompleteConsumer::Result Result;
  Results.MaybeAddResult(Result("short"));
  Results.MaybeAddResult(Result("long"));
  Results.MaybeAddResult(Result("signed"));
  Results.MaybeAddResult(Result("unsigned"));
  Results.MaybeAddResult(Result("void"));
  Results.MaybeAddResult(Result("char"));
  Results.MaybeAddResult(Result("int"));
  Results.MaybeAddResult(Result("float"));
  Results.MaybeAddResult(Result("double"));
  Results.MaybeAddResult(Result("enum"));
  Results.MaybeAddResult(Result("struct"));
  Results.MaybeAddResult(Result("union"));
  Results.MaybeAddResult(Result("const"));
  Results.MaybeAddResult(Result("volatile"));

  if (LangOpts.C99) {
    // C99-specific
    Results.MaybeAddResult(Result("_Complex"));
    Results.MaybeAddResult(Result("_Imaginary"));
    Results.MaybeAddResult(Result("_Bool"));
    Results.MaybeAddResult(Result("restrict"));
  }
  
  if (LangOpts.CPlusPlus) {
    // C++-specific
    Results.MaybeAddResult(Result("bool"));
    Results.MaybeAddResult(Result("class"));
    Results.MaybeAddResult(Result("wchar_t"));
    
    // typename qualified-id
    CodeCompletionString *Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("typename");
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddPlaceholderChunk("qualified-id");
    Results.MaybeAddResult(Result(Pattern));

    if (LangOpts.CPlusPlus0x) {
      Results.MaybeAddResult(Result("auto"));
      Results.MaybeAddResult(Result("char16_t"));
      Results.MaybeAddResult(Result("char32_t"));
      Results.MaybeAddResult(Result("decltype"));
    }
  }
  
  // GNU extensions
  if (LangOpts.GNUMode) {
    // FIXME: Enable when we actually support decimal floating point.
    //    Results.MaybeAddResult(Result("_Decimal32"));
    //    Results.MaybeAddResult(Result("_Decimal64"));
    //    Results.MaybeAddResult(Result("_Decimal128"));
    
    CodeCompletionString *Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("typeof");
    Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
    Pattern->AddPlaceholderChunk("expression-or-type");
    Pattern->AddChunk(CodeCompletionString::CK_RightParen);
    Results.MaybeAddResult(Result(Pattern));
  }
}

static void AddStorageSpecifiers(Action::CodeCompletionContext CCC,
                                 const LangOptions &LangOpts, 
                                 ResultBuilder &Results) {
  typedef CodeCompleteConsumer::Result Result;
  // Note: we don't suggest either "auto" or "register", because both
  // are pointless as storage specifiers. Elsewhere, we suggest "auto"
  // in C++0x as a type specifier.
  Results.MaybeAddResult(Result("extern"));
  Results.MaybeAddResult(Result("static"));
}

static void AddFunctionSpecifiers(Action::CodeCompletionContext CCC,
                                  const LangOptions &LangOpts, 
                                  ResultBuilder &Results) {
  typedef CodeCompleteConsumer::Result Result;
  switch (CCC) {
  case Action::CCC_Class:
  case Action::CCC_MemberTemplate:
    if (LangOpts.CPlusPlus) {
      Results.MaybeAddResult(Result("explicit"));
      Results.MaybeAddResult(Result("friend"));
      Results.MaybeAddResult(Result("mutable"));
      Results.MaybeAddResult(Result("virtual"));
    }    
    // Fall through

  case Action::CCC_ObjCInterface:
  case Action::CCC_ObjCImplementation:
  case Action::CCC_Namespace:
  case Action::CCC_Template:
    if (LangOpts.CPlusPlus || LangOpts.C99)
      Results.MaybeAddResult(Result("inline"));
    break;

  case Action::CCC_ObjCInstanceVariableList:
  case Action::CCC_Expression:
  case Action::CCC_Statement:
  case Action::CCC_ForInit:
  case Action::CCC_Condition:
    break;
  }
}

static void AddObjCExpressionResults(ResultBuilder &Results, bool NeedAt);
static void AddObjCStatementResults(ResultBuilder &Results, bool NeedAt);
static void AddObjCVisibilityResults(const LangOptions &LangOpts,
                                     ResultBuilder &Results,
                                     bool NeedAt);  
static void AddObjCImplementationResults(const LangOptions &LangOpts,
                                         ResultBuilder &Results,
                                         bool NeedAt);
static void AddObjCInterfaceResults(const LangOptions &LangOpts,
                                    ResultBuilder &Results,
                                    bool NeedAt);
static void AddObjCTopLevelResults(ResultBuilder &Results, bool NeedAt);

/// \brief Add language constructs that show up for "ordinary" names.
static void AddOrdinaryNameResults(Action::CodeCompletionContext CCC,
                                   Scope *S,
                                   Sema &SemaRef,
                                   ResultBuilder &Results) {
  typedef CodeCompleteConsumer::Result Result;
  switch (CCC) {
  case Action::CCC_Namespace:
    if (SemaRef.getLangOptions().CPlusPlus) {
      // namespace <identifier> { }
      CodeCompletionString *Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("namespace");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("identifier");
      Pattern->AddChunk(CodeCompletionString::CK_LeftBrace);
      Pattern->AddPlaceholderChunk("declarations");
      Pattern->AddChunk(CodeCompletionString::CK_VerticalSpace);
      Pattern->AddChunk(CodeCompletionString::CK_RightBrace);
      Results.MaybeAddResult(Result(Pattern));

      // namespace identifier = identifier ;
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("namespace");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("identifier");
      Pattern->AddChunk(CodeCompletionString::CK_Equal);
      Pattern->AddPlaceholderChunk("identifier");
      Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
      Results.MaybeAddResult(Result(Pattern));

      // Using directives
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("using");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddTextChunk("namespace");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("identifier");
      Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
      Results.MaybeAddResult(Result(Pattern));

      // asm(string-literal)      
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("asm");
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("string-literal");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
      Results.MaybeAddResult(Result(Pattern));

      // Explicit template instantiation
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("template");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("declaration");
      Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
      Results.MaybeAddResult(Result(Pattern));
    }
      
    if (SemaRef.getLangOptions().ObjC1)
      AddObjCTopLevelResults(Results, true);
      
    // Fall through

  case Action::CCC_Class:
    Results.MaybeAddResult(Result("typedef"));
    if (SemaRef.getLangOptions().CPlusPlus) {
      // Using declaration
      CodeCompletionString *Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("using");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("qualified-id");
      Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
      Results.MaybeAddResult(Result(Pattern));
      
      // using typename qualified-id; (only in a dependent context)
      if (SemaRef.CurContext->isDependentContext()) {
        Pattern = new CodeCompletionString;
        Pattern->AddTypedTextChunk("using");
        Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
        Pattern->AddTextChunk("typename");
        Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
        Pattern->AddPlaceholderChunk("qualified-id");
        Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
        Results.MaybeAddResult(Result(Pattern));
      }

      if (CCC == Action::CCC_Class) {
        // public:
        Pattern = new CodeCompletionString;
        Pattern->AddTypedTextChunk("public");
        Pattern->AddChunk(CodeCompletionString::CK_Colon);
        Results.MaybeAddResult(Result(Pattern));

        // protected:
        Pattern = new CodeCompletionString;
        Pattern->AddTypedTextChunk("protected");
        Pattern->AddChunk(CodeCompletionString::CK_Colon);
        Results.MaybeAddResult(Result(Pattern));

        // private:
        Pattern = new CodeCompletionString;
        Pattern->AddTypedTextChunk("private");
        Pattern->AddChunk(CodeCompletionString::CK_Colon);
        Results.MaybeAddResult(Result(Pattern));
      }
    }
    // Fall through

  case Action::CCC_Template:
  case Action::CCC_MemberTemplate:
    if (SemaRef.getLangOptions().CPlusPlus) {
      // template < parameters >
      CodeCompletionString *Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("template");
      Pattern->AddChunk(CodeCompletionString::CK_LeftAngle);
      Pattern->AddPlaceholderChunk("parameters");
      Pattern->AddChunk(CodeCompletionString::CK_RightAngle);
      Results.MaybeAddResult(Result(Pattern));
    }

    AddStorageSpecifiers(CCC, SemaRef.getLangOptions(), Results);
    AddFunctionSpecifiers(CCC, SemaRef.getLangOptions(), Results);
    break;

  case Action::CCC_ObjCInterface:
    AddObjCInterfaceResults(SemaRef.getLangOptions(), Results, true);
    AddStorageSpecifiers(CCC, SemaRef.getLangOptions(), Results);
    AddFunctionSpecifiers(CCC, SemaRef.getLangOptions(), Results);
    break;
      
  case Action::CCC_ObjCImplementation:
    AddObjCImplementationResults(SemaRef.getLangOptions(), Results, true);
    AddStorageSpecifiers(CCC, SemaRef.getLangOptions(), Results);
    AddFunctionSpecifiers(CCC, SemaRef.getLangOptions(), Results);
    break;
      
  case Action::CCC_ObjCInstanceVariableList:
    AddObjCVisibilityResults(SemaRef.getLangOptions(), Results, true);
    break;
      
  case Action::CCC_Statement: {
    Results.MaybeAddResult(Result("typedef"));

    CodeCompletionString *Pattern = 0;
    if (SemaRef.getLangOptions().CPlusPlus) {
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("try");
      Pattern->AddChunk(CodeCompletionString::CK_LeftBrace);
      Pattern->AddPlaceholderChunk("statements");
      Pattern->AddChunk(CodeCompletionString::CK_VerticalSpace);
      Pattern->AddChunk(CodeCompletionString::CK_RightBrace);
      Pattern->AddTextChunk("catch");
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("declaration");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Pattern->AddChunk(CodeCompletionString::CK_LeftBrace);
      Pattern->AddPlaceholderChunk("statements");
      Pattern->AddChunk(CodeCompletionString::CK_VerticalSpace);
      Pattern->AddChunk(CodeCompletionString::CK_RightBrace);
      Results.MaybeAddResult(Result(Pattern));
    }
    if (SemaRef.getLangOptions().ObjC1)
      AddObjCStatementResults(Results, true);
    
    // if (condition) { statements }
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("if");
    Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
    if (SemaRef.getLangOptions().CPlusPlus)
      Pattern->AddPlaceholderChunk("condition");
    else
      Pattern->AddPlaceholderChunk("expression");
    Pattern->AddChunk(CodeCompletionString::CK_RightParen);
    Pattern->AddChunk(CodeCompletionString::CK_LeftBrace);
    Pattern->AddPlaceholderChunk("statements");
    Pattern->AddChunk(CodeCompletionString::CK_VerticalSpace);
    Pattern->AddChunk(CodeCompletionString::CK_RightBrace);
    Results.MaybeAddResult(Result(Pattern));

    // switch (condition) { }
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("switch");
    Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
    if (SemaRef.getLangOptions().CPlusPlus)
      Pattern->AddPlaceholderChunk("condition");
    else
      Pattern->AddPlaceholderChunk("expression");
    Pattern->AddChunk(CodeCompletionString::CK_RightParen);
    Pattern->AddChunk(CodeCompletionString::CK_LeftBrace);
    Pattern->AddChunk(CodeCompletionString::CK_VerticalSpace);
    Pattern->AddChunk(CodeCompletionString::CK_RightBrace);
    Results.MaybeAddResult(Result(Pattern));

    // Switch-specific statements.
    if (!SemaRef.getSwitchStack().empty()) {
      // case expression:
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("case");
      Pattern->AddPlaceholderChunk("expression");
      Pattern->AddChunk(CodeCompletionString::CK_Colon);
      Results.MaybeAddResult(Result(Pattern));

      // default:
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("default");
      Pattern->AddChunk(CodeCompletionString::CK_Colon);
      Results.MaybeAddResult(Result(Pattern));
    }

    /// while (condition) { statements }
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("while");
    Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
    if (SemaRef.getLangOptions().CPlusPlus)
      Pattern->AddPlaceholderChunk("condition");
    else
      Pattern->AddPlaceholderChunk("expression");
    Pattern->AddChunk(CodeCompletionString::CK_RightParen);
    Pattern->AddChunk(CodeCompletionString::CK_LeftBrace);
    Pattern->AddPlaceholderChunk("statements");
    Pattern->AddChunk(CodeCompletionString::CK_VerticalSpace);
    Pattern->AddChunk(CodeCompletionString::CK_RightBrace);
    Results.MaybeAddResult(Result(Pattern));

    // do { statements } while ( expression );
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("do");
    Pattern->AddChunk(CodeCompletionString::CK_LeftBrace);
    Pattern->AddPlaceholderChunk("statements");
    Pattern->AddChunk(CodeCompletionString::CK_VerticalSpace);
    Pattern->AddChunk(CodeCompletionString::CK_RightBrace);
    Pattern->AddTextChunk("while");
    Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
    Pattern->AddPlaceholderChunk("expression");
    Pattern->AddChunk(CodeCompletionString::CK_RightParen);
    Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
    Results.MaybeAddResult(Result(Pattern));

    // for ( for-init-statement ; condition ; expression ) { statements }
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("for");
    Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
    if (SemaRef.getLangOptions().CPlusPlus || SemaRef.getLangOptions().C99)
      Pattern->AddPlaceholderChunk("init-statement");
    else
      Pattern->AddPlaceholderChunk("init-expression");
    Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
    Pattern->AddPlaceholderChunk("condition");
    Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
    Pattern->AddPlaceholderChunk("inc-expression");
    Pattern->AddChunk(CodeCompletionString::CK_RightParen);
    Pattern->AddChunk(CodeCompletionString::CK_LeftBrace);
    Pattern->AddPlaceholderChunk("statements");
    Pattern->AddChunk(CodeCompletionString::CK_VerticalSpace);
    Pattern->AddChunk(CodeCompletionString::CK_RightBrace);
    Results.MaybeAddResult(Result(Pattern));
    
    if (S->getContinueParent()) {
      // continue ;
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("continue");
      Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
      Results.MaybeAddResult(Result(Pattern));
    }

    if (S->getBreakParent()) {
      // break ;
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("break");
      Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
      Results.MaybeAddResult(Result(Pattern));
    }

    // "return expression ;" or "return ;", depending on whether we
    // know the function is void or not.
    bool isVoid = false;
    if (FunctionDecl *Function = dyn_cast<FunctionDecl>(SemaRef.CurContext))
      isVoid = Function->getResultType()->isVoidType();
    else if (ObjCMethodDecl *Method
                                 = dyn_cast<ObjCMethodDecl>(SemaRef.CurContext))
      isVoid = Method->getResultType()->isVoidType();
    else if (SemaRef.CurBlock && !SemaRef.CurBlock->ReturnType.isNull())
      isVoid = SemaRef.CurBlock->ReturnType->isVoidType();
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("return");
    if (!isVoid)
      Pattern->AddPlaceholderChunk("expression");
    Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
    Results.MaybeAddResult(Result(Pattern));

    // goto identifier ;
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("goto");
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddPlaceholderChunk("identifier");
    Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
    Results.MaybeAddResult(Result(Pattern));    

    // Using directives
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("using");
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddTextChunk("namespace");
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddPlaceholderChunk("identifier");
    Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
    Results.MaybeAddResult(Result(Pattern));
  }

  // Fall through (for statement expressions).
  case Action::CCC_ForInit:
  case Action::CCC_Condition:
    AddStorageSpecifiers(CCC, SemaRef.getLangOptions(), Results);
    // Fall through: conditions and statements can have expressions.

  case Action::CCC_Expression: {
    CodeCompletionString *Pattern = 0;
    if (SemaRef.getLangOptions().CPlusPlus) {
      // 'this', if we're in a non-static member function.
      if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(SemaRef.CurContext))
        if (!Method->isStatic())
          Results.MaybeAddResult(Result("this"));
      
      // true, false
      Results.MaybeAddResult(Result("true"));
      Results.MaybeAddResult(Result("false"));

      // dynamic_cast < type-id > ( expression )
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("dynamic_cast");
      Pattern->AddChunk(CodeCompletionString::CK_LeftAngle);
      Pattern->AddPlaceholderChunk("type-id");
      Pattern->AddChunk(CodeCompletionString::CK_RightAngle);
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("expression");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Results.MaybeAddResult(Result(Pattern));      
      
      // static_cast < type-id > ( expression )
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("static_cast");
      Pattern->AddChunk(CodeCompletionString::CK_LeftAngle);
      Pattern->AddPlaceholderChunk("type-id");
      Pattern->AddChunk(CodeCompletionString::CK_RightAngle);
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("expression");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Results.MaybeAddResult(Result(Pattern));      

      // reinterpret_cast < type-id > ( expression )
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("reinterpret_cast");
      Pattern->AddChunk(CodeCompletionString::CK_LeftAngle);
      Pattern->AddPlaceholderChunk("type-id");
      Pattern->AddChunk(CodeCompletionString::CK_RightAngle);
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("expression");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Results.MaybeAddResult(Result(Pattern));      

      // const_cast < type-id > ( expression )
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("const_cast");
      Pattern->AddChunk(CodeCompletionString::CK_LeftAngle);
      Pattern->AddPlaceholderChunk("type-id");
      Pattern->AddChunk(CodeCompletionString::CK_RightAngle);
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("expression");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Results.MaybeAddResult(Result(Pattern));      

      // typeid ( expression-or-type )
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("typeid");
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("expression-or-type");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Results.MaybeAddResult(Result(Pattern));      

      // new T ( ... )
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("new");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("type-id");
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("expressions");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Results.MaybeAddResult(Result(Pattern));      

      // new T [ ] ( ... )
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("new");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("type-id");
      Pattern->AddChunk(CodeCompletionString::CK_LeftBracket);
      Pattern->AddPlaceholderChunk("size");
      Pattern->AddChunk(CodeCompletionString::CK_RightBracket);
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("expressions");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Results.MaybeAddResult(Result(Pattern));      

      // delete expression
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("delete");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("expression");
      Results.MaybeAddResult(Result(Pattern));      

      // delete [] expression
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("delete");
      Pattern->AddChunk(CodeCompletionString::CK_LeftBracket);
      Pattern->AddChunk(CodeCompletionString::CK_RightBracket);
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("expression");
      Results.MaybeAddResult(Result(Pattern));

      // throw expression
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("throw");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("expression");
      Results.MaybeAddResult(Result(Pattern));
    }

    if (SemaRef.getLangOptions().ObjC1) {
      // Add "super", if we're in an Objective-C class with a superclass.
      if (ObjCMethodDecl *Method = SemaRef.getCurMethodDecl())
        if (Method->getClassInterface()->getSuperClass())
          Results.MaybeAddResult(Result("super"));
      
      AddObjCExpressionResults(Results, true);
    }

    // sizeof expression
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("sizeof");
    Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
    Pattern->AddPlaceholderChunk("expression-or-type");
    Pattern->AddChunk(CodeCompletionString::CK_RightParen);
    Results.MaybeAddResult(Result(Pattern));
    break;
  }
  }

  AddTypeSpecifierResults(SemaRef.getLangOptions(), Results);

  if (SemaRef.getLangOptions().CPlusPlus)
    Results.MaybeAddResult(Result("operator"));
}

/// \brief If the given declaration has an associated type, add it as a result 
/// type chunk.
static void AddResultTypeChunk(ASTContext &Context,
                               NamedDecl *ND,
                               CodeCompletionString *Result) {
  if (!ND)
    return;
  
  // Determine the type of the declaration (if it has a type).
  QualType T;
  if (FunctionDecl *Function = dyn_cast<FunctionDecl>(ND))
    T = Function->getResultType();
  else if (ObjCMethodDecl *Method = dyn_cast<ObjCMethodDecl>(ND))
    T = Method->getResultType();
  else if (FunctionTemplateDecl *FunTmpl = dyn_cast<FunctionTemplateDecl>(ND))
    T = FunTmpl->getTemplatedDecl()->getResultType();
  else if (EnumConstantDecl *Enumerator = dyn_cast<EnumConstantDecl>(ND))
    T = Context.getTypeDeclType(cast<TypeDecl>(Enumerator->getDeclContext()));
  else if (isa<UnresolvedUsingValueDecl>(ND)) {
    /* Do nothing: ignore unresolved using declarations*/
  } else if (ValueDecl *Value = dyn_cast<ValueDecl>(ND))
    T = Value->getType();
  else if (ObjCPropertyDecl *Property = dyn_cast<ObjCPropertyDecl>(ND))
    T = Property->getType();
  
  if (T.isNull() || Context.hasSameType(T, Context.DependentTy))
    return;
  
  std::string TypeStr;
  T.getAsStringInternal(TypeStr, Context.PrintingPolicy);
  Result->AddResultTypeChunk(TypeStr);
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
    CCStr->AddPlaceholderChunk(PlaceholderStr);
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
    CCStr->AddPlaceholderChunk(PlaceholderStr);
  }    
}

/// \brief Add a qualifier to the given code-completion string, if the
/// provided nested-name-specifier is non-NULL.
static void 
AddQualifierToCompletionString(CodeCompletionString *Result, 
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
    Result->AddInformativeChunk(PrintedNNS);
  else
    Result->AddTextChunk(PrintedNNS);
}

static void AddFunctionTypeQualsToCompletionString(CodeCompletionString *Result,
                                                   FunctionDecl *Function) {
  const FunctionProtoType *Proto
    = Function->getType()->getAs<FunctionProtoType>();
  if (!Proto || !Proto->getTypeQuals())
    return;

  std::string QualsStr;
  if (Proto->getTypeQuals() & Qualifiers::Const)
    QualsStr += " const";
  if (Proto->getTypeQuals() & Qualifiers::Volatile)
    QualsStr += " volatile";
  if (Proto->getTypeQuals() & Qualifiers::Restrict)
    QualsStr += " restrict";
  Result->AddInformativeChunk(QualsStr);
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
  
  if (Kind == RK_Pattern)
    return Pattern->Clone();
  
  CodeCompletionString *Result = new CodeCompletionString;

  if (Kind == RK_Keyword) {
    Result->AddTypedTextChunk(Keyword);
    return Result;
  }
  
  if (Kind == RK_Macro) {
    MacroInfo *MI = S.PP.getMacroInfo(Macro);
    assert(MI && "Not a macro?");

    Result->AddTypedTextChunk(Macro->getName());

    if (!MI->isFunctionLike())
      return Result;
    
    // Format a function-like macro with placeholders for the arguments.
    Result->AddChunk(Chunk(CodeCompletionString::CK_LeftParen));
    for (MacroInfo::arg_iterator A = MI->arg_begin(), AEnd = MI->arg_end();
         A != AEnd; ++A) {
      if (A != MI->arg_begin())
        Result->AddChunk(Chunk(CodeCompletionString::CK_Comma));
      
      if (!MI->isVariadic() || A != AEnd - 1) {
        // Non-variadic argument.
        Result->AddPlaceholderChunk((*A)->getName());
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
        Result->AddPlaceholderChunk(Arg);
      }
    }
    Result->AddChunk(Chunk(CodeCompletionString::CK_RightParen));
    return Result;
  }
  
  assert(Kind == RK_Declaration && "Missed a macro kind?");
  NamedDecl *ND = Declaration;
  
  if (StartsNestedNameSpecifier) {
    Result->AddTypedTextChunk(ND->getNameAsString());
    Result->AddTextChunk("::");
    return Result;
  }
  
  AddResultTypeChunk(S.Context, ND, Result);
  
  if (FunctionDecl *Function = dyn_cast<FunctionDecl>(ND)) {
    AddQualifierToCompletionString(Result, Qualifier, QualifierIsInformative, 
                                   S.Context);
    Result->AddTypedTextChunk(Function->getNameAsString());
    Result->AddChunk(Chunk(CodeCompletionString::CK_LeftParen));
    AddFunctionParameterChunks(S.Context, Function, Result);
    Result->AddChunk(Chunk(CodeCompletionString::CK_RightParen));
    AddFunctionTypeQualsToCompletionString(Result, Function);
    return Result;
  }
  
  if (FunctionTemplateDecl *FunTmpl = dyn_cast<FunctionTemplateDecl>(ND)) {
    AddQualifierToCompletionString(Result, Qualifier, QualifierIsInformative, 
                                   S.Context);
    FunctionDecl *Function = FunTmpl->getTemplatedDecl();
    Result->AddTypedTextChunk(Function->getNameAsString());
    
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
    AddFunctionTypeQualsToCompletionString(Result, Function);
    return Result;
  }
  
  if (TemplateDecl *Template = dyn_cast<TemplateDecl>(ND)) {
    AddQualifierToCompletionString(Result, Qualifier, QualifierIsInformative, 
                                   S.Context);
    Result->AddTypedTextChunk(Template->getNameAsString());
    Result->AddChunk(Chunk(CodeCompletionString::CK_LeftAngle));
    AddTemplateParameterChunks(S.Context, Template, Result);
    Result->AddChunk(Chunk(CodeCompletionString::CK_RightAngle));
    return Result;
  }
  
  if (ObjCMethodDecl *Method = dyn_cast<ObjCMethodDecl>(ND)) {
    Selector Sel = Method->getSelector();
    if (Sel.isUnarySelector()) {
      Result->AddTypedTextChunk(Sel.getIdentifierInfoForSlot(0)->getName());
      return Result;
    }

    std::string SelName = Sel.getIdentifierInfoForSlot(0)->getName().str();
    SelName += ':';
    if (StartParameter == 0)
      Result->AddTypedTextChunk(SelName);
    else {
      Result->AddInformativeChunk(SelName);
      
      // If there is only one parameter, and we're past it, add an empty
      // typed-text chunk since there is nothing to type.
      if (Method->param_size() == 1)
        Result->AddTypedTextChunk("");
    }
    unsigned Idx = 0;
    for (ObjCMethodDecl::param_iterator P = Method->param_begin(),
                                     PEnd = Method->param_end();
         P != PEnd; (void)++P, ++Idx) {
      if (Idx > 0) {
        std::string Keyword;
        if (Idx > StartParameter)
          Result->AddChunk(CodeCompletionString::CK_HorizontalSpace);
        if (IdentifierInfo *II = Sel.getIdentifierInfoForSlot(Idx))
          Keyword += II->getName().str();
        Keyword += ":";
        if (Idx < StartParameter || AllParametersAreInformative) {
          Result->AddInformativeChunk(Keyword);
        } else if (Idx == StartParameter)
          Result->AddTypedTextChunk(Keyword);
        else
          Result->AddTextChunk(Keyword);
      }
      
      // If we're before the starting parameter, skip the placeholder.
      if (Idx < StartParameter)
        continue;

      std::string Arg;
      (*P)->getType().getAsStringInternal(Arg, S.Context.PrintingPolicy);
      Arg = "(" + Arg + ")";
      if (IdentifierInfo *II = (*P)->getIdentifier())
        Arg += II->getName().str();
      if (AllParametersAreInformative)
        Result->AddInformativeChunk(Arg);
      else
        Result->AddPlaceholderChunk(Arg);
    }

    if (Method->isVariadic()) {
      if (AllParametersAreInformative)
        Result->AddInformativeChunk(", ...");
      else
        Result->AddPlaceholderChunk(", ...");
    }
    
    return Result;
  }

  if (Qualifier)
    AddQualifierToCompletionString(Result, Qualifier, QualifierIsInformative, 
                                   S.Context);

  Result->AddTypedTextChunk(ND->getNameAsString());
  return Result;
}

CodeCompletionString *
CodeCompleteConsumer::OverloadCandidate::CreateSignatureString(
                                                          unsigned CurrentArg,
                                                               Sema &S) const {
  typedef CodeCompletionString::Chunk Chunk;
  
  CodeCompletionString *Result = new CodeCompletionString;
  FunctionDecl *FDecl = getFunction();
  AddResultTypeChunk(S.Context, FDecl, Result);
  const FunctionProtoType *Proto 
    = dyn_cast<FunctionProtoType>(getFunctionType());
  if (!FDecl && !Proto) {
    // Function without a prototype. Just give the return type and a 
    // highlighted ellipsis.
    const FunctionType *FT = getFunctionType();
    Result->AddTextChunk(
            FT->getResultType().getAsString(S.Context.PrintingPolicy));
    Result->AddChunk(Chunk(CodeCompletionString::CK_LeftParen));
    Result->AddChunk(Chunk(CodeCompletionString::CK_CurrentParameter, "..."));
    Result->AddChunk(Chunk(CodeCompletionString::CK_RightParen));
    return Result;
  }
  
  if (FDecl)
    Result->AddTextChunk(FDecl->getNameAsString());
  else
    Result->AddTextChunk(
         Proto->getResultType().getAsString(S.Context.PrintingPolicy));
  
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
                             ArgString));
    else
      Result->AddTextChunk(ArgString);
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
      Selector XSel = X.getObjCSelector();
      Selector YSel = Y.getObjCSelector();
      if (!XSel.isNull() && !YSel.isNull()) {
        // We are comparing two selectors.
        unsigned N = std::min(XSel.getNumArgs(), YSel.getNumArgs());
        if (N == 0)
          ++N;
        for (unsigned I = 0; I != N; ++I) {
          IdentifierInfo *XId = XSel.getIdentifierInfoForSlot(I);
          IdentifierInfo *YId = YSel.getIdentifierInfoForSlot(I);
          if (!XId || !YId)
            return XId && !YId;
          
          switch (XId->getName().compare_lower(YId->getName())) {
          case -1: return true;
          case 1: return false;
          default: break;
          }
        }
    
        return XSel.getNumArgs() < YSel.getNumArgs();
      }

      // For non-selectors, order by kind.
      if (X.getNameKind() != Y.getNameKind())
        return X.getNameKind() < Y.getNameKind();
      
      // Order identifiers by comparison of their lowercased names.
      if (IdentifierInfo *XId = X.getAsIdentifierInfo())
        return XId->getName().compare_lower(
                                     Y.getAsIdentifierInfo()->getName()) < 0;

      // Order overloaded operators by the order in which they appear
      // in our list of operators.
      if (OverloadedOperatorKind XOp = X.getCXXOverloadedOperator())
        return XOp < Y.getCXXOverloadedOperator();

      // Order C++0x user-defined literal operators lexically by their
      // lowercased suffixes.
      if (IdentifierInfo *XLit = X.getCXXLiteralIdentifier())
        return XLit->getName().compare_lower(
                                  Y.getCXXLiteralIdentifier()->getName()) < 0;

      // The only stable ordering we have is to turn the name into a
      // string and then compare the lower-case strings. This is
      // inefficient, but thankfully does not happen too often.
      return llvm::StringRef(X.getAsString()).compare_lower(
                                                 Y.getAsString()) < 0;
    }
    
    /// \brief Retrieve the name that should be used to order a result.
    ///
    /// If the name needs to be constructed as a string, that string will be
    /// saved into Saved and the returned StringRef will refer to it.
    static llvm::StringRef getOrderedName(const Result &R,
                                          std::string &Saved) {
      switch (R.Kind) {
      case Result::RK_Keyword:
        return R.Keyword;
          
      case Result::RK_Pattern:
        return R.Pattern->getTypedText();
          
      case Result::RK_Macro:
        return R.Macro->getName();
          
      case Result::RK_Declaration:
        // Handle declarations below.
        break;
      }
            
      DeclarationName Name = R.Declaration->getDeclName();
      
      // If the name is a simple identifier (by far the common case), or a
      // zero-argument selector, just return a reference to that identifier.
      if (IdentifierInfo *Id = Name.getAsIdentifierInfo())
        return Id->getName();
      if (Name.isObjCZeroArgSelector())
        if (IdentifierInfo *Id
                          = Name.getObjCSelector().getIdentifierInfoForSlot(0))
          return Id->getName();
      
      Saved = Name.getAsString();
      return Saved;
    }
    
    bool operator()(const Result &X, const Result &Y) const {
      std::string XSaved, YSaved;
      llvm::StringRef XStr = getOrderedName(X, XSaved);
      llvm::StringRef YStr = getOrderedName(Y, YSaved);
      int cmp = XStr.compare_lower(YStr);
      if (cmp)
        return cmp < 0;
      
      // Non-hidden names precede hidden names.
      if (X.Hidden != Y.Hidden)
        return !X.Hidden;
      
      // Non-nested-name-specifiers precede nested-name-specifiers.
      if (X.StartsNestedNameSpecifier != Y.StartsNestedNameSpecifier)
        return !X.StartsNestedNameSpecifier;
      
      return false;
    }
  };
}

static void AddMacroResults(Preprocessor &PP, ResultBuilder &Results) {
  Results.EnterNewScope();
  for (Preprocessor::macro_iterator M = PP.macro_begin(), 
                                 MEnd = PP.macro_end();
       M != MEnd; ++M)
    Results.MaybeAddResult(M->first);
  Results.ExitScope();
}

static void HandleCodeCompleteResults(Sema *S,
                                      CodeCompleteConsumer *CodeCompleter,
                                     CodeCompleteConsumer::Result *Results,
                                     unsigned NumResults) {
  std::stable_sort(Results, Results + NumResults, SortCodeCompleteResult());

  if (CodeCompleter)
    CodeCompleter->ProcessCodeCompleteResults(*S, Results, NumResults);
  
  for (unsigned I = 0; I != NumResults; ++I)
    Results[I].Destroy();
}

void Sema::CodeCompleteOrdinaryName(Scope *S, 
                                    CodeCompletionContext CompletionContext) {
  typedef CodeCompleteConsumer::Result Result;
  ResultBuilder Results(*this);

  // Determine how to filter results, e.g., so that the names of
  // values (functions, enumerators, function templates, etc.) are
  // only allowed where we can have an expression.
  switch (CompletionContext) {
  case CCC_Namespace:
  case CCC_Class:
  case CCC_ObjCInterface:
  case CCC_ObjCImplementation:
  case CCC_ObjCInstanceVariableList:
  case CCC_Template:
  case CCC_MemberTemplate:
    Results.setFilter(&ResultBuilder::IsOrdinaryNonValueName);
    break;

  case CCC_Expression:
  case CCC_Statement:
  case CCC_ForInit:
  case CCC_Condition:
    Results.setFilter(&ResultBuilder::IsOrdinaryName);
    break;
  }

  CollectLookupResults(S, Context.getTranslationUnitDecl(), CurContext, 
                       Results);

  Results.EnterNewScope();
  AddOrdinaryNameResults(CompletionContext, S, *this, Results);
  Results.ExitScope();

  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, Results);
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

static void AddObjCProperties(ObjCContainerDecl *Container, 
                              bool AllowCategories,
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
      AddObjCProperties(*P, AllowCategories, CurContext, Results);
  } else if (ObjCInterfaceDecl *IFace = dyn_cast<ObjCInterfaceDecl>(Container)){
    if (AllowCategories) {
      // Look through categories.
      for (ObjCCategoryDecl *Category = IFace->getCategoryList();
           Category; Category = Category->getNextClassCategory())
        AddObjCProperties(Category, AllowCategories, CurContext, Results);
    }
    
    // Look through protocols.
    for (ObjCInterfaceDecl::protocol_iterator I = IFace->protocol_begin(),
                                              E = IFace->protocol_end(); 
         I != E; ++I)
      AddObjCProperties(*I, AllowCategories, CurContext, Results);
    
    // Look in the superclass.
    if (IFace->getSuperClass())
      AddObjCProperties(IFace->getSuperClass(), AllowCategories, CurContext, 
                        Results);
  } else if (const ObjCCategoryDecl *Category
                                    = dyn_cast<ObjCCategoryDecl>(Container)) {
    // Look through protocols.
    for (ObjCInterfaceDecl::protocol_iterator P = Category->protocol_begin(),
                                           PEnd = Category->protocol_end(); 
         P != PEnd; ++P)
      AddObjCProperties(*P, AllowCategories, CurContext, Results);
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
  Results.EnterNewScope();
  if (const RecordType *Record = BaseType->getAs<RecordType>()) {
    // Access to a C/C++ class, struct, or union.
    CollectMemberLookupResults(Record->getDecl(), Record->getDecl(), Results);

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
          Results.MaybeAddResult(Result("template"));
      }

      // We could have the start of a nested-name-specifier. Add those
      // results as well.
      // FIXME: We should really walk base classes to produce
      // nested-name-specifiers so that we produce more-precise results.
      Results.setFilter(&ResultBuilder::IsNestedNameSpecifier);
      CollectLookupResults(S, Context.getTranslationUnitDecl(), 
                           CurContext, Results);
    }
  } else if (!IsArrow && BaseType->getAsObjCInterfacePointerType()) {
    // Objective-C property reference.
    
    // Add property results based on our interface.
    const ObjCObjectPointerType *ObjCPtr
      = BaseType->getAsObjCInterfacePointerType();
    assert(ObjCPtr && "Non-NULL pointer guaranteed above!");
    AddObjCProperties(ObjCPtr->getInterfaceDecl(), true, CurContext, Results);
    
    // Add properties from the protocols in a qualified interface.
    for (ObjCObjectPointerType::qual_iterator I = ObjCPtr->qual_begin(),
                                              E = ObjCPtr->qual_end();
         I != E; ++I)
      AddObjCProperties(*I, true, CurContext, Results);
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
    AddMacroResults(PP, Results);

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
  CollectLookupResults(S, Context.getTranslationUnitDecl(), CurContext,Results);
  
  if (getLangOptions().CPlusPlus) {
    // We could have the start of a nested-name-specifier. Add those
    // results as well.
    Results.setFilter(&ResultBuilder::IsNestedNameSpecifier);
    CollectLookupResults(S, Context.getTranslationUnitDecl(), CurContext, 
                         Results);
  }
  
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, Results);
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
    
    Results.MaybeAddResult(CodeCompleteConsumer::Result(*E, Qualifier));
  }
  Results.ExitScope();
  
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, Results);
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

  // When we're code-completing for a call, we fall back to ordinary
  // name code-completion whenever we can't produce specific
  // results. We may want to revisit this strategy in the future,
  // e.g., by merging the two kinds of results.

  Expr *Fn = (Expr *)FnIn;
  Expr **Args = (Expr **)ArgsIn;

  // Ignore type-dependent call expressions entirely.
  if (Fn->isTypeDependent() || 
      Expr::hasAnyTypeDependentArguments(Args, NumArgs)) {
    CodeCompleteOrdinaryName(S, CCC_Expression);
    return;
  }

  // Build an overload candidate set based on the functions we find.
  OverloadCandidateSet CandidateSet;

  // FIXME: What if we're calling something that isn't a function declaration?
  // FIXME: What if we're calling a pseudo-destructor?
  // FIXME: What if we're calling a member function?
  
  Expr *NakedFn = Fn->IgnoreParenCasts();
  if (UnresolvedLookupExpr *ULE = dyn_cast<UnresolvedLookupExpr>(NakedFn))
    AddOverloadedCallCandidates(ULE, Args, NumArgs, CandidateSet,
                                /*PartialOverloading=*/ true);
  else if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(NakedFn)) {
    FunctionDecl *FDecl = dyn_cast<FunctionDecl>(DRE->getDecl());
    if (FDecl)
      AddOverloadCandidate(FDecl, Args, NumArgs, CandidateSet,
                           false, false, /*PartialOverloading*/ true);
  }
  
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

  if (Results.empty())
    CodeCompleteOrdinaryName(S, CCC_Expression);
  else
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

  // Try to instantiate any non-dependent declaration contexts before
  // we look in them.
  if (!isDependentScopeSpecifier(SS) && RequireCompleteDeclContext(SS))
    return;

  ResultBuilder Results(*this);
  CollectMemberLookupResults(Ctx, Ctx, Results);
  
  // The "template" keyword can follow "::" in the grammar, but only
  // put it into the grammar if the nested-name-specifier is dependent.
  NestedNameSpecifier *NNS = (NestedNameSpecifier *)SS.getScopeRep();
  if (!Results.empty() && NNS->isDependent())
    Results.MaybeAddResult("template");
  
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, Results);
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteUsing(Scope *S) {
  if (!CodeCompleter)
    return;
  
  ResultBuilder Results(*this, &ResultBuilder::IsNestedNameSpecifier);
  Results.EnterNewScope();
  
  // If we aren't in class scope, we could see the "namespace" keyword.
  if (!S->isClassScope())
    Results.MaybeAddResult(CodeCompleteConsumer::Result("namespace"));
  
  // After "using", we can see anything that would start a 
  // nested-name-specifier.
  CollectLookupResults(S, Context.getTranslationUnitDecl(), CurContext,Results);
  Results.ExitScope();
  
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, Results);
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteUsingDirective(Scope *S) {
  if (!CodeCompleter)
    return;
  
  // After "using namespace", we expect to see a namespace name or namespace
  // alias.
  ResultBuilder Results(*this, &ResultBuilder::IsNamespaceOrAlias);
  Results.EnterNewScope();
  CollectLookupResults(S, Context.getTranslationUnitDecl(), CurContext,Results);
  Results.ExitScope();
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, Results);
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
    AddMacroResults(PP, Results);
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteNamespaceAliasDecl(Scope *S)  {
  if (!CodeCompleter)
    return;
  
  // After "namespace", we expect to see a namespace or alias.
  ResultBuilder Results(*this, &ResultBuilder::IsNamespaceOrAlias);
  CollectLookupResults(S, Context.getTranslationUnitDecl(), CurContext,Results);
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, Results);
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
    Results.MaybeAddResult(Result(Spelling));
#include "clang/Basic/OperatorKinds.def"
  
  // Add any type names visible from the current scope
  CollectLookupResults(S, Context.getTranslationUnitDecl(), CurContext,Results);
  
  // Add any type specifiers
  AddTypeSpecifierResults(getLangOptions(), Results);
  
  // Add any nested-name-specifiers
  Results.setFilter(&ResultBuilder::IsNestedNameSpecifier);
  CollectLookupResults(S, Context.getTranslationUnitDecl(), CurContext,Results);
  Results.ExitScope();
  
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, Results);
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

// Macro that expands to @Keyword or Keyword, depending on whether NeedAt is
// true or false.
#define OBJC_AT_KEYWORD_NAME(NeedAt,Keyword) NeedAt? "@" #Keyword : #Keyword
static void AddObjCImplementationResults(const LangOptions &LangOpts,
                                         ResultBuilder &Results,
                                         bool NeedAt) {
  typedef CodeCompleteConsumer::Result Result;
  // Since we have an implementation, we can end it.
  Results.MaybeAddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,end)));
  
  CodeCompletionString *Pattern = 0;
  if (LangOpts.ObjC2) {
    // @dynamic
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,dynamic));
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddPlaceholderChunk("property");
    Results.MaybeAddResult(Result(Pattern));
    
    // @synthesize
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,synthesize));
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddPlaceholderChunk("property");
    Results.MaybeAddResult(Result(Pattern));
  }  
}

static void AddObjCInterfaceResults(const LangOptions &LangOpts,
                                    ResultBuilder &Results,
                                    bool NeedAt) {
  typedef CodeCompleteConsumer::Result Result;
  
  // Since we have an interface or protocol, we can end it.
  Results.MaybeAddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,end)));
  
  if (LangOpts.ObjC2) {
    // @property
    Results.MaybeAddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,property)));
  
    // @required
    Results.MaybeAddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,required)));
  
    // @optional
    Results.MaybeAddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,optional)));
  }
}

static void AddObjCTopLevelResults(ResultBuilder &Results, bool NeedAt) {
  typedef CodeCompleteConsumer::Result Result;
  CodeCompletionString *Pattern = 0;
  
  // @class name ;
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,class));
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("identifier");
  Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
  Results.MaybeAddResult(Result(Pattern));
  
  // @interface name 
  // FIXME: Could introduce the whole pattern, including superclasses and 
  // such.
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,interface));
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("class");
  Results.MaybeAddResult(Result(Pattern));
  
  // @protocol name
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,protocol));
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("protocol");
  Results.MaybeAddResult(Result(Pattern));
  
  // @implementation name
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,implementation));
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("class");
  Results.MaybeAddResult(Result(Pattern));
  
  // @compatibility_alias name
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,compatibility_alias));
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("alias");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("class");
  Results.MaybeAddResult(Result(Pattern));
}

void Sema::CodeCompleteObjCAtDirective(Scope *S, DeclPtrTy ObjCImpDecl,
                                       bool InInterface) {
  typedef CodeCompleteConsumer::Result Result;
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  if (ObjCImpDecl)
    AddObjCImplementationResults(getLangOptions(), Results, false);
  else if (InInterface)
    AddObjCInterfaceResults(getLangOptions(), Results, false);
  else
    AddObjCTopLevelResults(Results, false);
  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

static void AddObjCExpressionResults(ResultBuilder &Results, bool NeedAt) {
  typedef CodeCompleteConsumer::Result Result;
  CodeCompletionString *Pattern = 0;

  // @encode ( type-name )
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,encode));
  Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
  Pattern->AddPlaceholderChunk("type-name");
  Pattern->AddChunk(CodeCompletionString::CK_RightParen);
  Results.MaybeAddResult(Result(Pattern));
  
  // @protocol ( protocol-name )
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,protocol));
  Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
  Pattern->AddPlaceholderChunk("protocol-name");
  Pattern->AddChunk(CodeCompletionString::CK_RightParen);
  Results.MaybeAddResult(Result(Pattern));

  // @selector ( selector )
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,selector));
  Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
  Pattern->AddPlaceholderChunk("selector");
  Pattern->AddChunk(CodeCompletionString::CK_RightParen);
  Results.MaybeAddResult(Result(Pattern));
}

static void AddObjCStatementResults(ResultBuilder &Results, bool NeedAt) {
  typedef CodeCompleteConsumer::Result Result;
  CodeCompletionString *Pattern = 0;
  
  // @try { statements } @catch ( declaration ) { statements } @finally
  //   { statements }
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,try));
  Pattern->AddChunk(CodeCompletionString::CK_LeftBrace);
  Pattern->AddPlaceholderChunk("statements");
  Pattern->AddChunk(CodeCompletionString::CK_RightBrace);
  Pattern->AddTextChunk("@catch");
  Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
  Pattern->AddPlaceholderChunk("parameter");
  Pattern->AddChunk(CodeCompletionString::CK_RightParen);
  Pattern->AddChunk(CodeCompletionString::CK_LeftBrace);
  Pattern->AddPlaceholderChunk("statements");
  Pattern->AddChunk(CodeCompletionString::CK_RightBrace);
  Pattern->AddTextChunk("@finally");
  Pattern->AddChunk(CodeCompletionString::CK_LeftBrace);
  Pattern->AddPlaceholderChunk("statements");
  Pattern->AddChunk(CodeCompletionString::CK_RightBrace);
  Results.MaybeAddResult(Result(Pattern));
  
  // @throw
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,throw));
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("expression");
  Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
  Results.MaybeAddResult(Result(Pattern));
  
  // @synchronized ( expression ) { statements }
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,synchronized));
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
  Pattern->AddPlaceholderChunk("expression");
  Pattern->AddChunk(CodeCompletionString::CK_RightParen);
  Pattern->AddChunk(CodeCompletionString::CK_LeftBrace);
  Pattern->AddPlaceholderChunk("statements");
  Pattern->AddChunk(CodeCompletionString::CK_RightBrace);
  Results.MaybeAddResult(Result(Pattern));
}

static void AddObjCVisibilityResults(const LangOptions &LangOpts,
                                     ResultBuilder &Results,
                                     bool NeedAt) {
  typedef CodeCompleteConsumer::Result Result;
  Results.MaybeAddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,private)));
  Results.MaybeAddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,protected)));
  Results.MaybeAddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,public)));
  if (LangOpts.ObjC2)
    Results.MaybeAddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,package)));    
}

void Sema::CodeCompleteObjCAtVisibility(Scope *S) {
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  AddObjCVisibilityResults(getLangOptions(), Results, false);
  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteObjCAtStatement(Scope *S) {
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  AddObjCStatementResults(Results, false);
  AddObjCExpressionResults(Results, false);
  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteObjCAtExpression(Scope *S) {
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  AddObjCExpressionResults(Results, false);
  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

/// \brief Determine whether the addition of the given flag to an Objective-C
/// property's attributes will cause a conflict.
static bool ObjCPropertyFlagConflicts(unsigned Attributes, unsigned NewFlag) {
  // Check if we've already added this flag.
  if (Attributes & NewFlag)
    return true;
  
  Attributes |= NewFlag;
  
  // Check for collisions with "readonly".
  if ((Attributes & ObjCDeclSpec::DQ_PR_readonly) &&
      (Attributes & (ObjCDeclSpec::DQ_PR_readwrite |
                     ObjCDeclSpec::DQ_PR_assign |
                     ObjCDeclSpec::DQ_PR_copy |
                     ObjCDeclSpec::DQ_PR_retain)))
    return true;
  
  // Check for more than one of { assign, copy, retain }.
  unsigned AssignCopyRetMask = Attributes & (ObjCDeclSpec::DQ_PR_assign |
                                             ObjCDeclSpec::DQ_PR_copy |
                                             ObjCDeclSpec::DQ_PR_retain);
  if (AssignCopyRetMask &&
      AssignCopyRetMask != ObjCDeclSpec::DQ_PR_assign &&
      AssignCopyRetMask != ObjCDeclSpec::DQ_PR_copy &&
      AssignCopyRetMask != ObjCDeclSpec::DQ_PR_retain)
    return true;
  
  return false;
}

void Sema::CodeCompleteObjCPropertyFlags(Scope *S, ObjCDeclSpec &ODS) { 
  if (!CodeCompleter)
    return;
  
  unsigned Attributes = ODS.getPropertyAttributes();
  
  typedef CodeCompleteConsumer::Result Result;
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  if (!ObjCPropertyFlagConflicts(Attributes, ObjCDeclSpec::DQ_PR_readonly))
    Results.MaybeAddResult(CodeCompleteConsumer::Result("readonly"));
  if (!ObjCPropertyFlagConflicts(Attributes, ObjCDeclSpec::DQ_PR_assign))
    Results.MaybeAddResult(CodeCompleteConsumer::Result("assign"));
  if (!ObjCPropertyFlagConflicts(Attributes, ObjCDeclSpec::DQ_PR_readwrite))
    Results.MaybeAddResult(CodeCompleteConsumer::Result("readwrite"));
  if (!ObjCPropertyFlagConflicts(Attributes, ObjCDeclSpec::DQ_PR_retain))
    Results.MaybeAddResult(CodeCompleteConsumer::Result("retain"));
  if (!ObjCPropertyFlagConflicts(Attributes, ObjCDeclSpec::DQ_PR_copy))
    Results.MaybeAddResult(CodeCompleteConsumer::Result("copy"));
  if (!ObjCPropertyFlagConflicts(Attributes, ObjCDeclSpec::DQ_PR_nonatomic))
    Results.MaybeAddResult(CodeCompleteConsumer::Result("nonatomic"));
  if (!ObjCPropertyFlagConflicts(Attributes, ObjCDeclSpec::DQ_PR_setter)) {
    CodeCompletionString *Setter = new CodeCompletionString;
    Setter->AddTypedTextChunk("setter");
    Setter->AddTextChunk(" = ");
    Setter->AddPlaceholderChunk("method");
    Results.MaybeAddResult(CodeCompleteConsumer::Result(Setter));
  }
  if (!ObjCPropertyFlagConflicts(Attributes, ObjCDeclSpec::DQ_PR_getter)) {
    CodeCompletionString *Getter = new CodeCompletionString;
    Getter->AddTypedTextChunk("getter");
    Getter->AddTextChunk(" = ");
    Getter->AddPlaceholderChunk("method");
    Results.MaybeAddResult(CodeCompleteConsumer::Result(Getter));
  }
  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

/// \brief Descripts the kind of Objective-C method that we want to find
/// via code completion.
enum ObjCMethodKind {
  MK_Any, //< Any kind of method, provided it means other specified criteria.
  MK_ZeroArgSelector, //< Zero-argument (unary) selector.
  MK_OneArgSelector //< One-argument selector.
};

static bool isAcceptableObjCMethod(ObjCMethodDecl *Method,
                                   ObjCMethodKind WantKind,
                                   IdentifierInfo **SelIdents,
                                   unsigned NumSelIdents) {
  Selector Sel = Method->getSelector();
  if (NumSelIdents > Sel.getNumArgs())
    return false;
      
  switch (WantKind) {
  case MK_Any:             break;
  case MK_ZeroArgSelector: return Sel.isUnarySelector();
  case MK_OneArgSelector:  return Sel.getNumArgs() == 1;
  }

  for (unsigned I = 0; I != NumSelIdents; ++I)
    if (SelIdents[I] != Sel.getIdentifierInfoForSlot(I))
      return false;

  return true;
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
                           ObjCMethodKind WantKind,
                           IdentifierInfo **SelIdents,
                           unsigned NumSelIdents,
                           DeclContext *CurContext,
                           ResultBuilder &Results) {
  typedef CodeCompleteConsumer::Result Result;
  for (ObjCContainerDecl::method_iterator M = Container->meth_begin(),
                                       MEnd = Container->meth_end();
       M != MEnd; ++M) {
    if ((*M)->isInstanceMethod() == WantInstanceMethods) {
      // Check whether the selector identifiers we've been given are a 
      // subset of the identifiers for this particular method.
      if (!isAcceptableObjCMethod(*M, WantKind, SelIdents, NumSelIdents))
        continue;

      Result R = Result(*M, 0);
      R.StartParameter = NumSelIdents;
      R.AllParametersAreInformative = (WantKind != MK_Any);
      Results.MaybeAddResult(R, CurContext);
    }
  }
  
  ObjCInterfaceDecl *IFace = dyn_cast<ObjCInterfaceDecl>(Container);
  if (!IFace)
    return;
  
  // Add methods in protocols.
  const ObjCList<ObjCProtocolDecl> &Protocols= IFace->getReferencedProtocols();
  for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
                                            E = Protocols.end(); 
       I != E; ++I)
    AddObjCMethods(*I, WantInstanceMethods, WantKind, SelIdents, NumSelIdents, 
                   CurContext, Results);
  
  // Add methods in categories.
  for (ObjCCategoryDecl *CatDecl = IFace->getCategoryList(); CatDecl;
       CatDecl = CatDecl->getNextClassCategory()) {
    AddObjCMethods(CatDecl, WantInstanceMethods, WantKind, SelIdents, 
                   NumSelIdents, CurContext, Results);
    
    // Add a categories protocol methods.
    const ObjCList<ObjCProtocolDecl> &Protocols 
      = CatDecl->getReferencedProtocols();
    for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
                                              E = Protocols.end();
         I != E; ++I)
      AddObjCMethods(*I, WantInstanceMethods, WantKind, SelIdents, 
                     NumSelIdents, CurContext, Results);
    
    // Add methods in category implementations.
    if (ObjCCategoryImplDecl *Impl = CatDecl->getImplementation())
      AddObjCMethods(Impl, WantInstanceMethods, WantKind, SelIdents, 
                     NumSelIdents, CurContext, Results);
  }
  
  // Add methods in superclass.
  if (IFace->getSuperClass())
    AddObjCMethods(IFace->getSuperClass(), WantInstanceMethods, WantKind, 
                   SelIdents, NumSelIdents, CurContext, Results);

  // Add methods in our implementation, if any.
  if (ObjCImplementationDecl *Impl = IFace->getImplementation())
    AddObjCMethods(Impl, WantInstanceMethods, WantKind, SelIdents,
                   NumSelIdents, CurContext, Results);
}


void Sema::CodeCompleteObjCPropertyGetter(Scope *S, DeclPtrTy ClassDecl,
                                          DeclPtrTy *Methods,
                                          unsigned NumMethods) {
  typedef CodeCompleteConsumer::Result Result;

  // Try to find the interface where getters might live.
  ObjCInterfaceDecl *Class
    = dyn_cast_or_null<ObjCInterfaceDecl>(ClassDecl.getAs<Decl>());
  if (!Class) {
    if (ObjCCategoryDecl *Category
          = dyn_cast_or_null<ObjCCategoryDecl>(ClassDecl.getAs<Decl>()))
      Class = Category->getClassInterface();

    if (!Class)
      return;
  }

  // Find all of the potential getters.
  ResultBuilder Results(*this);
  Results.EnterNewScope();

  // FIXME: We need to do this because Objective-C methods don't get
  // pushed into DeclContexts early enough. Argh!
  for (unsigned I = 0; I != NumMethods; ++I) { 
    if (ObjCMethodDecl *Method
            = dyn_cast_or_null<ObjCMethodDecl>(Methods[I].getAs<Decl>()))
      if (Method->isInstanceMethod() &&
          isAcceptableObjCMethod(Method, MK_ZeroArgSelector, 0, 0)) {
        Result R = Result(Method, 0);
        R.AllParametersAreInformative = true;
        Results.MaybeAddResult(R, CurContext);
      }
  }

  AddObjCMethods(Class, true, MK_ZeroArgSelector, 0, 0, CurContext, Results);
  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter,Results.data(),Results.size());
}

void Sema::CodeCompleteObjCPropertySetter(Scope *S, DeclPtrTy ObjCImplDecl,
                                          DeclPtrTy *Methods,
                                          unsigned NumMethods) {
  typedef CodeCompleteConsumer::Result Result;

  // Try to find the interface where setters might live.
  ObjCInterfaceDecl *Class
    = dyn_cast_or_null<ObjCInterfaceDecl>(ObjCImplDecl.getAs<Decl>());
  if (!Class) {
    if (ObjCCategoryDecl *Category
          = dyn_cast_or_null<ObjCCategoryDecl>(ObjCImplDecl.getAs<Decl>()))
      Class = Category->getClassInterface();

    if (!Class)
      return;
  }

  // Find all of the potential getters.
  ResultBuilder Results(*this);
  Results.EnterNewScope();

  // FIXME: We need to do this because Objective-C methods don't get
  // pushed into DeclContexts early enough. Argh!
  for (unsigned I = 0; I != NumMethods; ++I) { 
    if (ObjCMethodDecl *Method
            = dyn_cast_or_null<ObjCMethodDecl>(Methods[I].getAs<Decl>()))
      if (Method->isInstanceMethod() &&
          isAcceptableObjCMethod(Method, MK_OneArgSelector, 0, 0)) {
        Result R = Result(Method, 0);
        R.AllParametersAreInformative = true;
        Results.MaybeAddResult(R, CurContext);
      }
  }

  AddObjCMethods(Class, true, MK_OneArgSelector, 0, 0, CurContext, Results);

  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter,Results.data(),Results.size());
}

void Sema::CodeCompleteObjCClassMessage(Scope *S, IdentifierInfo *FName,
                                        SourceLocation FNameLoc,
                                        IdentifierInfo **SelIdents,
                                        unsigned NumSelIdents) {
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
        return CodeCompleteObjCInstanceMessage(S, (Expr *)Super.get(),
                                               SelIdents, NumSelIdents);
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
    CXXScopeSpec SS;
    UnqualifiedId id;
    id.setIdentifier(FName, FNameLoc);
    OwningExprResult Super = ActOnIdExpression(S, SS, id, false, false);
    return CodeCompleteObjCInstanceMessage(S, (Expr *)Super.get(),
                                           SelIdents, NumSelIdents);
  }

  // Add all of the factory methods in this Objective-C class, its protocols,
  // superclasses, categories, implementation, etc.
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  AddObjCMethods(CDecl, false, MK_Any, SelIdents, NumSelIdents, CurContext, 
                 Results);  
  Results.ExitScope();
  
  // This also suppresses remaining diagnostics.
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteObjCInstanceMessage(Scope *S, ExprTy *Receiver,
                                           IdentifierInfo **SelIdents,
                                           unsigned NumSelIdents) {
  typedef CodeCompleteConsumer::Result Result;
  
  Expr *RecExpr = static_cast<Expr *>(Receiver);
  
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
        AddObjCMethods(ClassDecl, false, MK_Any, SelIdents, NumSelIdents, 
                       CurContext, Results);
    }
  } 
  // Handle messages to a qualified ID ("id<foo>").
  else if (const ObjCObjectPointerType *QualID
             = ReceiverType->getAsObjCQualifiedIdType()) {
    // Search protocols for instance methods.
    for (ObjCObjectPointerType::qual_iterator I = QualID->qual_begin(),
                                              E = QualID->qual_end(); 
         I != E; ++I)
      AddObjCMethods(*I, true, MK_Any, SelIdents, NumSelIdents, CurContext, 
                     Results);
  }
  // Handle messages to a pointer to interface type.
  else if (const ObjCObjectPointerType *IFacePtr
                              = ReceiverType->getAsObjCInterfacePointerType()) {
    // Search the class, its superclasses, etc., for instance methods.
    AddObjCMethods(IFacePtr->getInterfaceDecl(), true, MK_Any, SelIdents,
                   NumSelIdents, CurContext, Results);
    
    // Search protocols for instance methods.
    for (ObjCObjectPointerType::qual_iterator I = IFacePtr->qual_begin(),
         E = IFacePtr->qual_end(); 
         I != E; ++I)
      AddObjCMethods(*I, true, MK_Any, SelIdents, NumSelIdents, CurContext, 
                     Results);
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

/// \brief Add all of the Objective-C interface declarations that we find in
/// the given (translation unit) context.
static void AddInterfaceResults(DeclContext *Ctx, DeclContext *CurContext,
                                bool OnlyForwardDeclarations,
                                bool OnlyUnimplemented,
                                ResultBuilder &Results) {
  typedef CodeCompleteConsumer::Result Result;
  
  for (DeclContext::decl_iterator D = Ctx->decls_begin(), 
                               DEnd = Ctx->decls_end();
       D != DEnd; ++D) {
    // Record any interfaces we find.
    if (ObjCInterfaceDecl *Class = dyn_cast<ObjCInterfaceDecl>(*D))
      if ((!OnlyForwardDeclarations || Class->isForwardDecl()) &&
          (!OnlyUnimplemented || !Class->getImplementation()))
        Results.MaybeAddResult(Result(Class, 0), CurContext);

    // Record any forward-declared interfaces we find.
    if (ObjCClassDecl *Forward = dyn_cast<ObjCClassDecl>(*D)) {
      for (ObjCClassDecl::iterator C = Forward->begin(), CEnd = Forward->end();
           C != CEnd; ++C)
        if ((!OnlyForwardDeclarations || C->getInterface()->isForwardDecl()) &&
            (!OnlyUnimplemented || !C->getInterface()->getImplementation()))
          Results.MaybeAddResult(Result(C->getInterface(), 0), CurContext);
    }
  }
}

void Sema::CodeCompleteObjCInterfaceDecl(Scope *S) { 
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  
  // Add all classes.
  AddInterfaceResults(Context.getTranslationUnitDecl(), CurContext, true,
                      false, Results);

  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteObjCSuperclass(Scope *S, IdentifierInfo *ClassName) { 
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  
  // Make sure that we ignore the class we're currently defining.
  NamedDecl *CurClass
    = LookupSingleName(TUScope, ClassName, LookupOrdinaryName);
  if (CurClass && isa<ObjCInterfaceDecl>(CurClass))
    Results.Ignore(CurClass);

  // Add all classes.
  AddInterfaceResults(Context.getTranslationUnitDecl(), CurContext, false,
                      false, Results);

  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteObjCImplementationDecl(Scope *S) { 
  ResultBuilder Results(*this);
  Results.EnterNewScope();

  // Add all unimplemented classes.
  AddInterfaceResults(Context.getTranslationUnitDecl(), CurContext, false,
                      true, Results);

  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}

void Sema::CodeCompleteObjCInterfaceCategory(Scope *S, 
                                             IdentifierInfo *ClassName) {
  typedef CodeCompleteConsumer::Result Result;
  
  ResultBuilder Results(*this);
  
  // Ignore any categories we find that have already been implemented by this
  // interface.
  llvm::SmallPtrSet<IdentifierInfo *, 16> CategoryNames;
  NamedDecl *CurClass
    = LookupSingleName(TUScope, ClassName, LookupOrdinaryName);
  if (ObjCInterfaceDecl *Class = dyn_cast_or_null<ObjCInterfaceDecl>(CurClass))
    for (ObjCCategoryDecl *Category = Class->getCategoryList(); Category;
         Category = Category->getNextClassCategory())
      CategoryNames.insert(Category->getIdentifier());
  
  // Add all of the categories we know about.
  Results.EnterNewScope();
  TranslationUnitDecl *TU = Context.getTranslationUnitDecl();
  for (DeclContext::decl_iterator D = TU->decls_begin(), 
                               DEnd = TU->decls_end();
       D != DEnd; ++D) 
    if (ObjCCategoryDecl *Category = dyn_cast<ObjCCategoryDecl>(*D))
      if (CategoryNames.insert(Category->getIdentifier()))
          Results.MaybeAddResult(Result(Category, 0), CurContext);
  Results.ExitScope();
  
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());  
}

void Sema::CodeCompleteObjCImplementationCategory(Scope *S, 
                                                  IdentifierInfo *ClassName) {
  typedef CodeCompleteConsumer::Result Result;
  
  // Find the corresponding interface. If we couldn't find the interface, the
  // program itself is ill-formed. However, we'll try to be helpful still by
  // providing the list of all of the categories we know about.
  NamedDecl *CurClass
    = LookupSingleName(TUScope, ClassName, LookupOrdinaryName);
  ObjCInterfaceDecl *Class = dyn_cast_or_null<ObjCInterfaceDecl>(CurClass);
  if (!Class)
    return CodeCompleteObjCInterfaceCategory(S, ClassName);
    
  ResultBuilder Results(*this);
  
  // Add all of the categories that have have corresponding interface 
  // declarations in this class and any of its superclasses, except for
  // already-implemented categories in the class itself.
  llvm::SmallPtrSet<IdentifierInfo *, 16> CategoryNames;
  Results.EnterNewScope();
  bool IgnoreImplemented = true;
  while (Class) {
    for (ObjCCategoryDecl *Category = Class->getCategoryList(); Category;
         Category = Category->getNextClassCategory())
      if ((!IgnoreImplemented || !Category->getImplementation()) &&
          CategoryNames.insert(Category->getIdentifier()))
        Results.MaybeAddResult(Result(Category, 0), CurContext);
    
    Class = Class->getSuperClass();
    IgnoreImplemented = false;
  }
  Results.ExitScope();
  
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());  
}

void Sema::CodeCompleteObjCPropertyDefinition(Scope *S, DeclPtrTy ObjCImpDecl) {
  typedef CodeCompleteConsumer::Result Result;
  ResultBuilder Results(*this);

  // Figure out where this @synthesize lives.
  ObjCContainerDecl *Container
    = dyn_cast_or_null<ObjCContainerDecl>(ObjCImpDecl.getAs<Decl>());
  if (!Container || 
      (!isa<ObjCImplementationDecl>(Container) && 
       !isa<ObjCCategoryImplDecl>(Container)))
    return; 

  // Ignore any properties that have already been implemented.
  for (DeclContext::decl_iterator D = Container->decls_begin(), 
                               DEnd = Container->decls_end();
       D != DEnd; ++D)
    if (ObjCPropertyImplDecl *PropertyImpl = dyn_cast<ObjCPropertyImplDecl>(*D))
      Results.Ignore(PropertyImpl->getPropertyDecl());
  
  // Add any properties that we find.
  Results.EnterNewScope();
  if (ObjCImplementationDecl *ClassImpl
        = dyn_cast<ObjCImplementationDecl>(Container))
    AddObjCProperties(ClassImpl->getClassInterface(), false, CurContext, 
                      Results);
  else
    AddObjCProperties(cast<ObjCCategoryImplDecl>(Container)->getCategoryDecl(),
                      false, CurContext, Results);
  Results.ExitScope();
  
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());  
}

void Sema::CodeCompleteObjCPropertySynthesizeIvar(Scope *S, 
                                                  IdentifierInfo *PropertyName,
                                                  DeclPtrTy ObjCImpDecl) {
  typedef CodeCompleteConsumer::Result Result;
  ResultBuilder Results(*this);

  // Figure out where this @synthesize lives.
  ObjCContainerDecl *Container
    = dyn_cast_or_null<ObjCContainerDecl>(ObjCImpDecl.getAs<Decl>());
  if (!Container || 
      (!isa<ObjCImplementationDecl>(Container) && 
       !isa<ObjCCategoryImplDecl>(Container)))
    return; 
  
  // Figure out which interface we're looking into.
  ObjCInterfaceDecl *Class = 0;
  if (ObjCImplementationDecl *ClassImpl
                                 = dyn_cast<ObjCImplementationDecl>(Container))  
    Class = ClassImpl->getClassInterface();
  else
    Class = cast<ObjCCategoryImplDecl>(Container)->getCategoryDecl()
                                                          ->getClassInterface();

  // Add all of the instance variables in this class and its superclasses.
  Results.EnterNewScope();
  for(; Class; Class = Class->getSuperClass()) {
    // FIXME: We could screen the type of each ivar for compatibility with
    // the property, but is that being too paternal?
    for (ObjCInterfaceDecl::ivar_iterator IVar = Class->ivar_begin(),
                                       IVarEnd = Class->ivar_end();
         IVar != IVarEnd; ++IVar) 
      Results.MaybeAddResult(Result(*IVar, 0), CurContext);
  }
  Results.ExitScope();
  
  HandleCodeCompleteResults(this, CodeCompleter, Results.data(),Results.size());
}
