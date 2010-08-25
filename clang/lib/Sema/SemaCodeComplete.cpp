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
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include <list>
#include <map>
#include <vector>

using namespace clang;
using namespace sema;

namespace {
  /// \brief A container of code-completion results.
  class ResultBuilder {
  public:
    /// \brief The type of a name-lookup filter, which can be provided to the
    /// name-lookup routines to specify which declarations should be included in
    /// the result set (when it returns true) and which declarations should be
    /// filtered out (returns false).
    typedef bool (ResultBuilder::*LookupFilter)(NamedDecl *) const;
    
    typedef CodeCompletionResult Result;
    
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

    /// \brief Whether we should allow declarations as
    /// nested-name-specifiers that would otherwise be filtered out.
    bool AllowNestedNameSpecifiers;

    /// \brief If set, the type that we would prefer our resulting value
    /// declarations to have.
    ///
    /// Closely matching the preferred type gives a boost to a result's 
    /// priority.
    CanQualType PreferredType;
    
    /// \brief A list of shadow maps, which is used to model name hiding at
    /// different levels of, e.g., the inheritance hierarchy.
    std::list<ShadowMap> ShadowMaps;
    
    void AdjustResultPriorityForPreferredType(Result &R);

  public:
    explicit ResultBuilder(Sema &SemaRef, LookupFilter Filter = 0)
      : SemaRef(SemaRef), Filter(Filter), AllowNestedNameSpecifiers(false) { }
    
    /// \brief Whether we should include code patterns in the completion
    /// results.
    bool includeCodePatterns() const {
      return SemaRef.CodeCompleter && 
      SemaRef.CodeCompleter->includeCodePatterns();
    }
    
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
    
    /// \brief Specify the preferred type.
    void setPreferredType(QualType T) { 
      PreferredType = SemaRef.Context.getCanonicalType(T); 
    }
    
    /// \brief Specify whether nested-name-specifiers are allowed.
    void allowNestedNameSpecifiers(bool Allow = true) {
      AllowNestedNameSpecifiers = Allow;
    }

    /// \brief Determine whether the given declaration is at all interesting
    /// as a code-completion result.
    ///
    /// \param ND the declaration that we are inspecting.
    ///
    /// \param AsNestedNameSpecifier will be set true if this declaration is
    /// only interesting when it is a nested-name-specifier.
    bool isInterestingDecl(NamedDecl *ND, bool &AsNestedNameSpecifier) const;
    
    /// \brief Check whether the result is hidden by the Hiding declaration.
    ///
    /// \returns true if the result is hidden and cannot be found, false if
    /// the hidden result could still be found. When false, \p R may be
    /// modified to describe how the result can be found (e.g., via extra
    /// qualification).
    bool CheckHiddenResult(Result &R, DeclContext *CurContext,
                           NamedDecl *Hiding);
    
    /// \brief Add a new result to this result set (if it isn't already in one
    /// of the shadow maps), or replace an existing result (for, e.g., a 
    /// redeclaration).
    ///
    /// \param CurContext the result to add (if it is unique).
    ///
    /// \param R the context in which this result will be named.
    void MaybeAddResult(Result R, DeclContext *CurContext = 0);
    
    /// \brief Add a new result to this result set, where we already know
    /// the hiding declation (if any).
    ///
    /// \param R the result to add (if it is unique).
    ///
    /// \param CurContext the context in which this result will be named.
    ///
    /// \param Hiding the declaration that hides the result.
    ///
    /// \param InBaseClass whether the result was found in a base
    /// class of the searched context.
    void AddResult(Result R, DeclContext *CurContext, NamedDecl *Hiding,
                   bool InBaseClass);
    
    /// \brief Add a new non-declaration result to this result set.
    void AddResult(Result R);

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
    bool IsOrdinaryNonTypeName(NamedDecl *ND) const;
    bool IsIntegralConstantValue(NamedDecl *ND) const;
    bool IsOrdinaryNonValueName(NamedDecl *ND) const;
    bool IsNestedNameSpecifier(NamedDecl *ND) const;
    bool IsEnum(NamedDecl *ND) const;
    bool IsClassOrStruct(NamedDecl *ND) const;
    bool IsUnion(NamedDecl *ND) const;
    bool IsNamespace(NamedDecl *ND) const;
    bool IsNamespaceOrAlias(NamedDecl *ND) const;
    bool IsType(NamedDecl *ND) const;
    bool IsMember(NamedDecl *ND) const;
    bool IsObjCIvar(NamedDecl *ND) const;
    bool IsObjCMessageReceiver(NamedDecl *ND) const;
    bool IsObjCCollection(NamedDecl *ND) const;
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
    
    if (NamespaceDecl *Namespace = dyn_cast<NamespaceDecl>(Parent)) {
      if (!Namespace->getIdentifier())
        continue;

      Result = NestedNameSpecifier::Create(Context, Result, Namespace);
    }
    else if (TagDecl *TD = dyn_cast<TagDecl>(Parent))
      Result = NestedNameSpecifier::Create(Context, Result,
                                           false,
                                     Context.getTypeDeclType(TD).getTypePtr());
  }  
  return Result;
}

bool ResultBuilder::isInterestingDecl(NamedDecl *ND, 
                                      bool &AsNestedNameSpecifier) const {
  AsNestedNameSpecifier = false;

  ND = ND->getUnderlyingDecl();
  unsigned IDNS = ND->getIdentifierNamespace();

  // Skip unnamed entities.
  if (!ND->getDeclName())
    return false;
  
  // Friend declarations and declarations introduced due to friends are never
  // added as results.
  if (IDNS & (Decl::IDNS_OrdinaryFriend | Decl::IDNS_TagFriend))
    return false;
  
  // Class template (partial) specializations are never added as results.
  if (isa<ClassTemplateSpecializationDecl>(ND) ||
      isa<ClassTemplatePartialSpecializationDecl>(ND))
    return false;
  
  // Using declarations themselves are never added as results.
  if (isa<UsingDecl>(ND))
    return false;
  
  // Some declarations have reserved names that we don't want to ever show.
  if (const IdentifierInfo *Id = ND->getIdentifier()) {
    // __va_list_tag is a freak of nature. Find it and skip it.
    if (Id->isStr("__va_list_tag") || Id->isStr("__builtin_va_list"))
      return false;
    
    // Filter out names reserved for the implementation (C99 7.1.3, 
    // C++ [lib.global.names]) if they come from a system header.
    //
    // FIXME: Add predicate for this.
    if (Id->getLength() >= 2) {
      const char *Name = Id->getNameStart();
      if (Name[0] == '_' &&
          (Name[1] == '_' || (Name[1] >= 'A' && Name[1] <= 'Z')) &&
          (ND->getLocation().isInvalid() ||
           SemaRef.SourceMgr.isInSystemHeader(
                          SemaRef.SourceMgr.getSpellingLoc(ND->getLocation()))))
        return false;
    }
  }
 
  // C++ constructors are never found by name lookup.
  if (isa<CXXConstructorDecl>(ND))
    return false;
  
  if (Filter == &ResultBuilder::IsNestedNameSpecifier ||
      ((isa<NamespaceDecl>(ND) || isa<NamespaceAliasDecl>(ND)) &&
       Filter != &ResultBuilder::IsNamespace &&
       Filter != &ResultBuilder::IsNamespaceOrAlias))
    AsNestedNameSpecifier = true;

  // Filter out any unwanted results.
  if (Filter && !(this->*Filter)(ND)) {
    // Check whether it is interesting as a nested-name-specifier.
    if (AllowNestedNameSpecifiers && SemaRef.getLangOptions().CPlusPlus && 
        IsNestedNameSpecifier(ND) &&
        (Filter != &ResultBuilder::IsMember ||
         (isa<CXXRecordDecl>(ND) && 
          cast<CXXRecordDecl>(ND)->isInjectedClassName()))) {
      AsNestedNameSpecifier = true;
      return true;
    }

    return false;
  }  
  // ... then it must be interesting!
  return true;
}

bool ResultBuilder::CheckHiddenResult(Result &R, DeclContext *CurContext,
                                      NamedDecl *Hiding) {
  // In C, there is no way to refer to a hidden name.
  // FIXME: This isn't true; we can find a tag name hidden by an ordinary
  // name if we introduce the tag type.
  if (!SemaRef.getLangOptions().CPlusPlus)
    return true;
  
  DeclContext *HiddenCtx = R.Declaration->getDeclContext()->getLookupContext();
  
  // There is no way to qualify a name declared in a function or method.
  if (HiddenCtx->isFunctionOrMethod())
    return true;
  
  if (HiddenCtx == Hiding->getDeclContext()->getLookupContext())
    return true;
  
  // We can refer to the result with the appropriate qualification. Do it.
  R.Hidden = true;
  R.QualifierIsInformative = false;
  
  if (!R.Qualifier)
    R.Qualifier = getRequiredQualification(SemaRef.Context, 
                                           CurContext, 
                                           R.Declaration->getDeclContext());
  return false;
}

/// \brief A simplified classification of types used to determine whether two
/// types are "similar enough" when adjusting priorities.
SimplifiedTypeClass clang::getSimplifiedTypeClass(CanQualType T) {
  switch (T->getTypeClass()) {
  case Type::Builtin:
    switch (cast<BuiltinType>(T)->getKind()) {
      case BuiltinType::Void:
        return STC_Void;
        
      case BuiltinType::NullPtr:
        return STC_Pointer;
        
      case BuiltinType::Overload:
      case BuiltinType::Dependent:
      case BuiltinType::UndeducedAuto:
        return STC_Other;
        
      case BuiltinType::ObjCId:
      case BuiltinType::ObjCClass:
      case BuiltinType::ObjCSel:
        return STC_ObjectiveC;
        
      default:
        return STC_Arithmetic;
    }
    return STC_Other;
    
  case Type::Complex:
    return STC_Arithmetic;
    
  case Type::Pointer:
    return STC_Pointer;
    
  case Type::BlockPointer:
    return STC_Block;
    
  case Type::LValueReference:
  case Type::RValueReference:
    return getSimplifiedTypeClass(T->getAs<ReferenceType>()->getPointeeType());
    
  case Type::ConstantArray:
  case Type::IncompleteArray:
  case Type::VariableArray:
  case Type::DependentSizedArray:
    return STC_Array;
    
  case Type::DependentSizedExtVector:
  case Type::Vector:
  case Type::ExtVector:
    return STC_Arithmetic;
    
  case Type::FunctionProto:
  case Type::FunctionNoProto:
    return STC_Function;
    
  case Type::Record:
    return STC_Record;
    
  case Type::Enum:
    return STC_Arithmetic;
    
  case Type::ObjCObject:
  case Type::ObjCInterface:
  case Type::ObjCObjectPointer:
    return STC_ObjectiveC;
    
  default:
    return STC_Other;
  }
}

/// \brief Get the type that a given expression will have if this declaration
/// is used as an expression in its "typical" code-completion form.
QualType clang::getDeclUsageType(ASTContext &C, NamedDecl *ND) {
  ND = cast<NamedDecl>(ND->getUnderlyingDecl());
  
  if (TypeDecl *Type = dyn_cast<TypeDecl>(ND))
    return C.getTypeDeclType(Type);
  if (ObjCInterfaceDecl *Iface = dyn_cast<ObjCInterfaceDecl>(ND))
    return C.getObjCInterfaceType(Iface);
  
  QualType T;
  if (FunctionDecl *Function = dyn_cast<FunctionDecl>(ND))
    T = Function->getCallResultType();
  else if (ObjCMethodDecl *Method = dyn_cast<ObjCMethodDecl>(ND))
    T = Method->getSendResultType();
  else if (FunctionTemplateDecl *FunTmpl = dyn_cast<FunctionTemplateDecl>(ND))
    T = FunTmpl->getTemplatedDecl()->getCallResultType();
  else if (EnumConstantDecl *Enumerator = dyn_cast<EnumConstantDecl>(ND))
    T = C.getTypeDeclType(cast<EnumDecl>(Enumerator->getDeclContext()));
  else if (ObjCPropertyDecl *Property = dyn_cast<ObjCPropertyDecl>(ND))
    T = Property->getType();
  else if (ValueDecl *Value = dyn_cast<ValueDecl>(ND))
    T = Value->getType();
  else
    return QualType();
  
  return T.getNonReferenceType();
}

void ResultBuilder::AdjustResultPriorityForPreferredType(Result &R) {
  QualType T = getDeclUsageType(SemaRef.Context, R.Declaration);
  if (T.isNull())
    return;
  
  CanQualType TC = SemaRef.Context.getCanonicalType(T);
  // Check for exactly-matching types (modulo qualifiers).
  if (SemaRef.Context.hasSameUnqualifiedType(PreferredType, TC)) {
    if (PreferredType->isVoidType())
      R.Priority += CCD_VoidMatch;
    else
      R.Priority /= CCF_ExactTypeMatch;
  } // Check for nearly-matching types, based on classification of each.
  else if ((getSimplifiedTypeClass(PreferredType)
                                               == getSimplifiedTypeClass(TC)) &&
           !(PreferredType->isEnumeralType() && TC->isEnumeralType()))
    R.Priority /= CCF_SimilarTypeMatch;  
}

void ResultBuilder::MaybeAddResult(Result R, DeclContext *CurContext) {
  assert(!ShadowMaps.empty() && "Must enter into a results scope");
  
  if (R.Kind != Result::RK_Declaration) {
    // For non-declaration results, just add the result.
    Results.push_back(R);
    return;
  }

  // Look through using declarations.
  if (UsingShadowDecl *Using = dyn_cast<UsingShadowDecl>(R.Declaration)) {
    MaybeAddResult(Result(Using->getTargetDecl(), R.Qualifier), CurContext);
    return;
  }
  
  Decl *CanonDecl = R.Declaration->getCanonicalDecl();
  unsigned IDNS = CanonDecl->getIdentifierNamespace();

  bool AsNestedNameSpecifier = false;
  if (!isInterestingDecl(R.Declaration, AsNestedNameSpecifier))
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
      if (I->first->hasTagIdentifierNamespace() &&
          (IDNS & (Decl::IDNS_Member | Decl::IDNS_Ordinary | 
                   Decl::IDNS_ObjCProtocol)))
        continue;
      
      // Protocols are in distinct namespaces from everything else.
      if (((I->first->getIdentifierNamespace() & Decl::IDNS_ObjCProtocol)
           || (IDNS & Decl::IDNS_ObjCProtocol)) &&
          I->first->getIdentifierNamespace() != IDNS)
        continue;
      
      // The newly-added result is hidden by an entry in the shadow map.
      if (CheckHiddenResult(R, CurContext, I->first))
        return;
      
      break;
    }
  }
  
  // Make sure that any given declaration only shows up in the result set once.
  if (!AllDeclsFound.insert(CanonDecl))
    return;
  
  // If the filter is for nested-name-specifiers, then this result starts a
  // nested-name-specifier.
  if (AsNestedNameSpecifier) {
    R.StartsNestedNameSpecifier = true;
    R.Priority = CCP_NestedNameSpecifier;
  } else if (!PreferredType.isNull())
      AdjustResultPriorityForPreferredType(R);

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

void ResultBuilder::AddResult(Result R, DeclContext *CurContext, 
                              NamedDecl *Hiding, bool InBaseClass = false) {
  if (R.Kind != Result::RK_Declaration) {
    // For non-declaration results, just add the result.
    Results.push_back(R);
    return;
  }

  // Look through using declarations.
  if (UsingShadowDecl *Using = dyn_cast<UsingShadowDecl>(R.Declaration)) {
    AddResult(Result(Using->getTargetDecl(), R.Qualifier), CurContext, Hiding);
    return;
  }
  
  bool AsNestedNameSpecifier = false;
  if (!isInterestingDecl(R.Declaration, AsNestedNameSpecifier))
    return;
  
  if (Hiding && CheckHiddenResult(R, CurContext, Hiding))
    return;
      
  // Make sure that any given declaration only shows up in the result set once.
  if (!AllDeclsFound.insert(R.Declaration->getCanonicalDecl()))
    return;
  
  // If the filter is for nested-name-specifiers, then this result starts a
  // nested-name-specifier.
  if (AsNestedNameSpecifier) {
    R.StartsNestedNameSpecifier = true;
    R.Priority = CCP_NestedNameSpecifier;
  }
  else if (Filter == &ResultBuilder::IsMember && !R.Qualifier && InBaseClass &&
           isa<CXXRecordDecl>(R.Declaration->getDeclContext()
                                                  ->getLookupContext()))
    R.QualifierIsInformative = true;

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
  
  // Adjust the priority if this result comes from a base class.
  if (InBaseClass)
    R.Priority += CCD_InBaseClass;
  
  if (!PreferredType.isNull())
    AdjustResultPriorityForPreferredType(R);
  
  // Insert this result into the set of results.
  Results.push_back(R);
}

void ResultBuilder::AddResult(Result R) {
  assert(R.Kind != Result::RK_Declaration && 
          "Declaration results need more context");
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
  ND = cast<NamedDecl>(ND->getUnderlyingDecl());

  unsigned IDNS = Decl::IDNS_Ordinary;
  if (SemaRef.getLangOptions().CPlusPlus)
    IDNS |= Decl::IDNS_Tag | Decl::IDNS_Namespace | Decl::IDNS_Member;
  else if (SemaRef.getLangOptions().ObjC1 && isa<ObjCIvarDecl>(ND))
    return true;

  return ND->getIdentifierNamespace() & IDNS;
}

/// \brief Determines whether this given declaration will be found by
/// ordinary name lookup but is not a type name.
bool ResultBuilder::IsOrdinaryNonTypeName(NamedDecl *ND) const {
  ND = cast<NamedDecl>(ND->getUnderlyingDecl());
  if (isa<TypeDecl>(ND) || isa<ObjCInterfaceDecl>(ND))
    return false;
  
  unsigned IDNS = Decl::IDNS_Ordinary;
  if (SemaRef.getLangOptions().CPlusPlus)
    IDNS |= Decl::IDNS_Tag | Decl::IDNS_Namespace | Decl::IDNS_Member;
  else if (SemaRef.getLangOptions().ObjC1 && isa<ObjCIvarDecl>(ND))
    return true;
  
  return ND->getIdentifierNamespace() & IDNS;
}

bool ResultBuilder::IsIntegralConstantValue(NamedDecl *ND) const {
  if (!IsOrdinaryNonTypeName(ND))
    return 0;
  
  if (ValueDecl *VD = dyn_cast<ValueDecl>(ND->getUnderlyingDecl()))
    if (VD->getType()->isIntegralOrEnumerationType())
      return true;
        
  return false;
}

/// \brief Determines whether this given declaration will be found by
/// ordinary name lookup.
bool ResultBuilder::IsOrdinaryNonValueName(NamedDecl *ND) const {
  ND = cast<NamedDecl>(ND->getUnderlyingDecl());

  unsigned IDNS = Decl::IDNS_Ordinary;
  if (SemaRef.getLangOptions().CPlusPlus)
    IDNS |= Decl::IDNS_Tag | Decl::IDNS_Namespace;
  
  return (ND->getIdentifierNamespace() & IDNS) && 
    !isa<ValueDecl>(ND) && !isa<FunctionTemplateDecl>(ND) && 
    !isa<ObjCPropertyDecl>(ND);
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
    return RD->getTagKind() == TTK_Class ||
    RD->getTagKind() == TTK_Struct;
  
  return false;
}

/// \brief Determines whether the given declaration is a union.
bool ResultBuilder::IsUnion(NamedDecl *ND) const {
  // Allow us to find class templates, too.
  if (ClassTemplateDecl *ClassTemplate = dyn_cast<ClassTemplateDecl>(ND))
    ND = ClassTemplate->getTemplatedDecl();
  
  if (RecordDecl *RD = dyn_cast<RecordDecl>(ND))
    return RD->getTagKind() == TTK_Union;
  
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
  if (UsingShadowDecl *Using = dyn_cast<UsingShadowDecl>(ND))
    ND = Using->getTargetDecl();
  
  return isa<TypeDecl>(ND) || isa<ObjCInterfaceDecl>(ND);
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

static bool isObjCReceiverType(ASTContext &C, QualType T) {
  T = C.getCanonicalType(T);
  switch (T->getTypeClass()) {
  case Type::ObjCObject: 
  case Type::ObjCInterface:
  case Type::ObjCObjectPointer:
    return true;
      
  case Type::Builtin:
    switch (cast<BuiltinType>(T)->getKind()) {
    case BuiltinType::ObjCId:
    case BuiltinType::ObjCClass:
    case BuiltinType::ObjCSel:
      return true;
      
    default:
      break;
    }
    return false;
      
  default:
    break;
  }
  
  if (!C.getLangOptions().CPlusPlus)
    return false;

  // FIXME: We could perform more analysis here to determine whether a 
  // particular class type has any conversions to Objective-C types. For now,
  // just accept all class types.
  return T->isDependentType() || T->isRecordType();
}

bool ResultBuilder::IsObjCMessageReceiver(NamedDecl *ND) const {
  QualType T = getDeclUsageType(SemaRef.Context, ND);
  if (T.isNull())
    return false;
  
  T = SemaRef.Context.getBaseElementType(T);
  return isObjCReceiverType(SemaRef.Context, T);
}

bool ResultBuilder::IsObjCCollection(NamedDecl *ND) const {
  if ((SemaRef.getLangOptions().CPlusPlus && !IsOrdinaryName(ND)) ||
      (!SemaRef.getLangOptions().CPlusPlus && !IsOrdinaryNonTypeName(ND)))
    return false;
  
  QualType T = getDeclUsageType(SemaRef.Context, ND);
  if (T.isNull())
    return false;
  
  T = SemaRef.Context.getBaseElementType(T);
  return T->isObjCObjectType() || T->isObjCObjectPointerType() ||
         T->isObjCIdType() || 
         (SemaRef.getLangOptions().CPlusPlus && T->isRecordType());
}

/// \rief Determines whether the given declaration is an Objective-C
/// instance variable.
bool ResultBuilder::IsObjCIvar(NamedDecl *ND) const {
  return isa<ObjCIvarDecl>(ND);
}

namespace {
  /// \brief Visible declaration consumer that adds a code-completion result
  /// for each visible declaration.
  class CodeCompletionDeclConsumer : public VisibleDeclConsumer {
    ResultBuilder &Results;
    DeclContext *CurContext;
    
  public:
    CodeCompletionDeclConsumer(ResultBuilder &Results, DeclContext *CurContext)
      : Results(Results), CurContext(CurContext) { }
    
    virtual void FoundDecl(NamedDecl *ND, NamedDecl *Hiding, bool InBaseClass) {
      Results.AddResult(ND, CurContext, Hiding, InBaseClass);
    }
  };
}

/// \brief Add type specifiers for the current language as keyword results.
static void AddTypeSpecifierResults(const LangOptions &LangOpts,
                                    ResultBuilder &Results) {
  typedef CodeCompletionResult Result;
  Results.AddResult(Result("short", CCP_Type));
  Results.AddResult(Result("long", CCP_Type));
  Results.AddResult(Result("signed", CCP_Type));
  Results.AddResult(Result("unsigned", CCP_Type));
  Results.AddResult(Result("void", CCP_Type));
  Results.AddResult(Result("char", CCP_Type));
  Results.AddResult(Result("int", CCP_Type));
  Results.AddResult(Result("float", CCP_Type));
  Results.AddResult(Result("double", CCP_Type));
  Results.AddResult(Result("enum", CCP_Type));
  Results.AddResult(Result("struct", CCP_Type));
  Results.AddResult(Result("union", CCP_Type));
  Results.AddResult(Result("const", CCP_Type));
  Results.AddResult(Result("volatile", CCP_Type));

  if (LangOpts.C99) {
    // C99-specific
    Results.AddResult(Result("_Complex", CCP_Type));
    Results.AddResult(Result("_Imaginary", CCP_Type));
    Results.AddResult(Result("_Bool", CCP_Type));
    Results.AddResult(Result("restrict", CCP_Type));
  }
  
  if (LangOpts.CPlusPlus) {
    // C++-specific
    Results.AddResult(Result("bool", CCP_Type));
    Results.AddResult(Result("class", CCP_Type));
    Results.AddResult(Result("wchar_t", CCP_Type));
    
    // typename qualified-id
    CodeCompletionString *Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("typename");
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddPlaceholderChunk("qualifier");
    Pattern->AddTextChunk("::");
    Pattern->AddPlaceholderChunk("name");
    Results.AddResult(Result(Pattern));
    
    if (LangOpts.CPlusPlus0x) {
      Results.AddResult(Result("auto", CCP_Type));
      Results.AddResult(Result("char16_t", CCP_Type));
      Results.AddResult(Result("char32_t", CCP_Type));
      
      CodeCompletionString *Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("decltype");
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("expression");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Results.AddResult(Result(Pattern));
    }
  }
  
  // GNU extensions
  if (LangOpts.GNUMode) {
    // FIXME: Enable when we actually support decimal floating point.
    //    Results.AddResult(Result("_Decimal32"));
    //    Results.AddResult(Result("_Decimal64"));
    //    Results.AddResult(Result("_Decimal128"));
    
    CodeCompletionString *Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("typeof");
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddPlaceholderChunk("expression");
    Results.AddResult(Result(Pattern));

    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("typeof");
    Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
    Pattern->AddPlaceholderChunk("type");
    Pattern->AddChunk(CodeCompletionString::CK_RightParen);
    Results.AddResult(Result(Pattern));
  }
}

static void AddStorageSpecifiers(Action::ParserCompletionContext CCC,
                                 const LangOptions &LangOpts, 
                                 ResultBuilder &Results) {
  typedef CodeCompletionResult Result;
  // Note: we don't suggest either "auto" or "register", because both
  // are pointless as storage specifiers. Elsewhere, we suggest "auto"
  // in C++0x as a type specifier.
  Results.AddResult(Result("extern"));
  Results.AddResult(Result("static"));
}

static void AddFunctionSpecifiers(Action::ParserCompletionContext CCC,
                                  const LangOptions &LangOpts, 
                                  ResultBuilder &Results) {
  typedef CodeCompletionResult Result;
  switch (CCC) {
  case Action::PCC_Class:
  case Action::PCC_MemberTemplate:
    if (LangOpts.CPlusPlus) {
      Results.AddResult(Result("explicit"));
      Results.AddResult(Result("friend"));
      Results.AddResult(Result("mutable"));
      Results.AddResult(Result("virtual"));
    }    
    // Fall through

  case Action::PCC_ObjCInterface:
  case Action::PCC_ObjCImplementation:
  case Action::PCC_Namespace:
  case Action::PCC_Template:
    if (LangOpts.CPlusPlus || LangOpts.C99)
      Results.AddResult(Result("inline"));
    break;

  case Action::PCC_ObjCInstanceVariableList:
  case Action::PCC_Expression:
  case Action::PCC_Statement:
  case Action::PCC_ForInit:
  case Action::PCC_Condition:
  case Action::PCC_RecoveryInFunction:
  case Action::PCC_Type:
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

static void AddTypedefResult(ResultBuilder &Results) {
  CodeCompletionString *Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("typedef");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("type");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("name");
  Results.AddResult(CodeCompletionResult(Pattern));        
}

static bool WantTypesInContext(Action::ParserCompletionContext CCC,
                               const LangOptions &LangOpts) {
  if (LangOpts.CPlusPlus)
    return true;
  
  switch (CCC) {
  case Action::PCC_Namespace:
  case Action::PCC_Class:
  case Action::PCC_ObjCInstanceVariableList:
  case Action::PCC_Template:
  case Action::PCC_MemberTemplate:
  case Action::PCC_Statement:
  case Action::PCC_RecoveryInFunction:
  case Action::PCC_Type:
    return true;
    
  case Action::PCC_ObjCInterface:
  case Action::PCC_ObjCImplementation:
  case Action::PCC_Expression:
  case Action::PCC_Condition:
    return false;
    
  case Action::PCC_ForInit:
    return LangOpts.ObjC1 || LangOpts.C99;
  }
  
  return false;
}

/// \brief Add language constructs that show up for "ordinary" names.
static void AddOrdinaryNameResults(Action::ParserCompletionContext CCC,
                                   Scope *S,
                                   Sema &SemaRef,
                                   ResultBuilder &Results) {
  typedef CodeCompletionResult Result;
  switch (CCC) {
  case Action::PCC_Namespace:
    if (SemaRef.getLangOptions().CPlusPlus) {
      CodeCompletionString *Pattern = 0;
      
      if (Results.includeCodePatterns()) {
        // namespace <identifier> { declarations }
        CodeCompletionString *Pattern = new CodeCompletionString;
        Pattern->AddTypedTextChunk("namespace");
        Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
        Pattern->AddPlaceholderChunk("identifier");
        Pattern->AddChunk(CodeCompletionString::CK_LeftBrace);
        Pattern->AddPlaceholderChunk("declarations");
        Pattern->AddChunk(CodeCompletionString::CK_VerticalSpace);
        Pattern->AddChunk(CodeCompletionString::CK_RightBrace);
        Results.AddResult(Result(Pattern));
      }
      
      // namespace identifier = identifier ;
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("namespace");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("name");
      Pattern->AddChunk(CodeCompletionString::CK_Equal);
      Pattern->AddPlaceholderChunk("namespace");
      Results.AddResult(Result(Pattern));

      // Using directives
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("using");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddTextChunk("namespace");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("identifier");
      Results.AddResult(Result(Pattern));

      // asm(string-literal)      
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("asm");
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("string-literal");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Results.AddResult(Result(Pattern));

      if (Results.includeCodePatterns()) {
        // Explicit template instantiation
        Pattern = new CodeCompletionString;
        Pattern->AddTypedTextChunk("template");
        Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
        Pattern->AddPlaceholderChunk("declaration");
        Results.AddResult(Result(Pattern));
      }
    }
      
    if (SemaRef.getLangOptions().ObjC1)
      AddObjCTopLevelResults(Results, true);
      
    AddTypedefResult(Results);
    // Fall through

  case Action::PCC_Class:
    if (SemaRef.getLangOptions().CPlusPlus) {
      // Using declaration
      CodeCompletionString *Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("using");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("qualifier");
      Pattern->AddTextChunk("::");
      Pattern->AddPlaceholderChunk("name");
      Results.AddResult(Result(Pattern));
      
      // using typename qualifier::name (only in a dependent context)
      if (SemaRef.CurContext->isDependentContext()) {
        Pattern = new CodeCompletionString;
        Pattern->AddTypedTextChunk("using");
        Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
        Pattern->AddTextChunk("typename");
        Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
        Pattern->AddPlaceholderChunk("qualifier");
        Pattern->AddTextChunk("::");
        Pattern->AddPlaceholderChunk("name");
        Results.AddResult(Result(Pattern));
      }

      if (CCC == Action::PCC_Class) {
        AddTypedefResult(Results);

        // public:
        Pattern = new CodeCompletionString;
        Pattern->AddTypedTextChunk("public");
        Pattern->AddChunk(CodeCompletionString::CK_Colon);
        Results.AddResult(Result(Pattern));

        // protected:
        Pattern = new CodeCompletionString;
        Pattern->AddTypedTextChunk("protected");
        Pattern->AddChunk(CodeCompletionString::CK_Colon);
        Results.AddResult(Result(Pattern));

        // private:
        Pattern = new CodeCompletionString;
        Pattern->AddTypedTextChunk("private");
        Pattern->AddChunk(CodeCompletionString::CK_Colon);
        Results.AddResult(Result(Pattern));
      }
    }
    // Fall through

  case Action::PCC_Template:
  case Action::PCC_MemberTemplate:
    if (SemaRef.getLangOptions().CPlusPlus && Results.includeCodePatterns()) {
      // template < parameters >
      CodeCompletionString *Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("template");
      Pattern->AddChunk(CodeCompletionString::CK_LeftAngle);
      Pattern->AddPlaceholderChunk("parameters");
      Pattern->AddChunk(CodeCompletionString::CK_RightAngle);
      Results.AddResult(Result(Pattern));
    }

    AddStorageSpecifiers(CCC, SemaRef.getLangOptions(), Results);
    AddFunctionSpecifiers(CCC, SemaRef.getLangOptions(), Results);
    break;

  case Action::PCC_ObjCInterface:
    AddObjCInterfaceResults(SemaRef.getLangOptions(), Results, true);
    AddStorageSpecifiers(CCC, SemaRef.getLangOptions(), Results);
    AddFunctionSpecifiers(CCC, SemaRef.getLangOptions(), Results);
    break;
      
  case Action::PCC_ObjCImplementation:
    AddObjCImplementationResults(SemaRef.getLangOptions(), Results, true);
    AddStorageSpecifiers(CCC, SemaRef.getLangOptions(), Results);
    AddFunctionSpecifiers(CCC, SemaRef.getLangOptions(), Results);
    break;
      
  case Action::PCC_ObjCInstanceVariableList:
    AddObjCVisibilityResults(SemaRef.getLangOptions(), Results, true);
    break;
      
  case Action::PCC_RecoveryInFunction:
  case Action::PCC_Statement: {
    AddTypedefResult(Results);

    CodeCompletionString *Pattern = 0;
    if (SemaRef.getLangOptions().CPlusPlus && Results.includeCodePatterns()) {
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
      Results.AddResult(Result(Pattern));
    }
    if (SemaRef.getLangOptions().ObjC1)
      AddObjCStatementResults(Results, true);
    
    if (Results.includeCodePatterns()) {
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
      Results.AddResult(Result(Pattern));

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
      Results.AddResult(Result(Pattern));
    }
    
    // Switch-specific statements.
    if (!SemaRef.getCurFunction()->SwitchStack.empty()) {
      // case expression:
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("case");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("expression");
      Pattern->AddChunk(CodeCompletionString::CK_Colon);
      Results.AddResult(Result(Pattern));

      // default:
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("default");
      Pattern->AddChunk(CodeCompletionString::CK_Colon);
      Results.AddResult(Result(Pattern));
    }

    if (Results.includeCodePatterns()) {
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
      Results.AddResult(Result(Pattern));

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
      Results.AddResult(Result(Pattern));

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
      Results.AddResult(Result(Pattern));
    }
    
    if (S->getContinueParent()) {
      // continue ;
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("continue");
      Results.AddResult(Result(Pattern));
    }

    if (S->getBreakParent()) {
      // break ;
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("break");
      Results.AddResult(Result(Pattern));
    }

    // "return expression ;" or "return ;", depending on whether we
    // know the function is void or not.
    bool isVoid = false;
    if (FunctionDecl *Function = dyn_cast<FunctionDecl>(SemaRef.CurContext))
      isVoid = Function->getResultType()->isVoidType();
    else if (ObjCMethodDecl *Method
                                 = dyn_cast<ObjCMethodDecl>(SemaRef.CurContext))
      isVoid = Method->getResultType()->isVoidType();
    else if (SemaRef.getCurBlock() && 
             !SemaRef.getCurBlock()->ReturnType.isNull())
      isVoid = SemaRef.getCurBlock()->ReturnType->isVoidType();
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("return");
    if (!isVoid) {
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("expression");
    }
    Results.AddResult(Result(Pattern));

    // goto identifier ;
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("goto");
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddPlaceholderChunk("label");
    Results.AddResult(Result(Pattern));    

    // Using directives
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("using");
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddTextChunk("namespace");
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddPlaceholderChunk("identifier");
    Results.AddResult(Result(Pattern));
  }

  // Fall through (for statement expressions).
  case Action::PCC_ForInit:
  case Action::PCC_Condition:
    AddStorageSpecifiers(CCC, SemaRef.getLangOptions(), Results);
    // Fall through: conditions and statements can have expressions.

  case Action::PCC_Expression: {
    CodeCompletionString *Pattern = 0;
    if (SemaRef.getLangOptions().CPlusPlus) {
      // 'this', if we're in a non-static member function.
      if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(SemaRef.CurContext))
        if (!Method->isStatic())
          Results.AddResult(Result("this"));
      
      // true, false
      Results.AddResult(Result("true"));
      Results.AddResult(Result("false"));

      // dynamic_cast < type-id > ( expression )
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("dynamic_cast");
      Pattern->AddChunk(CodeCompletionString::CK_LeftAngle);
      Pattern->AddPlaceholderChunk("type");
      Pattern->AddChunk(CodeCompletionString::CK_RightAngle);
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("expression");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Results.AddResult(Result(Pattern));      
      
      // static_cast < type-id > ( expression )
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("static_cast");
      Pattern->AddChunk(CodeCompletionString::CK_LeftAngle);
      Pattern->AddPlaceholderChunk("type");
      Pattern->AddChunk(CodeCompletionString::CK_RightAngle);
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("expression");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Results.AddResult(Result(Pattern));      

      // reinterpret_cast < type-id > ( expression )
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("reinterpret_cast");
      Pattern->AddChunk(CodeCompletionString::CK_LeftAngle);
      Pattern->AddPlaceholderChunk("type");
      Pattern->AddChunk(CodeCompletionString::CK_RightAngle);
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("expression");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Results.AddResult(Result(Pattern));      

      // const_cast < type-id > ( expression )
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("const_cast");
      Pattern->AddChunk(CodeCompletionString::CK_LeftAngle);
      Pattern->AddPlaceholderChunk("type");
      Pattern->AddChunk(CodeCompletionString::CK_RightAngle);
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("expression");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Results.AddResult(Result(Pattern));      

      // typeid ( expression-or-type )
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("typeid");
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("expression-or-type");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Results.AddResult(Result(Pattern));      

      // new T ( ... )
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("new");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("type");
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("expressions");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Results.AddResult(Result(Pattern));      

      // new T [ ] ( ... )
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("new");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("type");
      Pattern->AddChunk(CodeCompletionString::CK_LeftBracket);
      Pattern->AddPlaceholderChunk("size");
      Pattern->AddChunk(CodeCompletionString::CK_RightBracket);
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddPlaceholderChunk("expressions");
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      Results.AddResult(Result(Pattern));      

      // delete expression
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("delete");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("expression");
      Results.AddResult(Result(Pattern));      

      // delete [] expression
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("delete");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddChunk(CodeCompletionString::CK_LeftBracket);
      Pattern->AddChunk(CodeCompletionString::CK_RightBracket);
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("expression");
      Results.AddResult(Result(Pattern));

      // throw expression
      Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk("throw");
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddPlaceholderChunk("expression");
      Results.AddResult(Result(Pattern));
      
      // FIXME: Rethrow?
    }

    if (SemaRef.getLangOptions().ObjC1) {
      // Add "super", if we're in an Objective-C class with a superclass.
      if (ObjCMethodDecl *Method = SemaRef.getCurMethodDecl()) {
        // The interface can be NULL.
        if (ObjCInterfaceDecl *ID = Method->getClassInterface())
          if (ID->getSuperClass())
            Results.AddResult(Result("super"));
      }

      AddObjCExpressionResults(Results, true);
    }

    // sizeof expression
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("sizeof");
    Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
    Pattern->AddPlaceholderChunk("expression-or-type");
    Pattern->AddChunk(CodeCompletionString::CK_RightParen);
    Results.AddResult(Result(Pattern));
    break;
  }
      
  case Action::PCC_Type:
    break;
  }

  if (WantTypesInContext(CCC, SemaRef.getLangOptions()))
    AddTypeSpecifierResults(SemaRef.getLangOptions(), Results);

  if (SemaRef.getLangOptions().CPlusPlus && CCC != Action::PCC_Type)
    Results.AddResult(Result("operator"));
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
  
  PrintingPolicy Policy(Context.PrintingPolicy);
  Policy.AnonymousTagLocations = false;
  
  std::string TypeStr;
  T.getAsStringInternal(TypeStr, Policy);
  Result->AddResultTypeChunk(TypeStr);
}

static void MaybeAddSentinel(ASTContext &Context, NamedDecl *FunctionOrMethod,
                             CodeCompletionString *Result) {
  if (SentinelAttr *Sentinel = FunctionOrMethod->getAttr<SentinelAttr>())
    if (Sentinel->getSentinel() == 0) {
      if (Context.getLangOptions().ObjC1 &&
          Context.Idents.get("nil").hasMacroDefinition())
        Result->AddTextChunk(", nil");
      else if (Context.Idents.get("NULL").hasMacroDefinition())
        Result->AddTextChunk(", NULL");
      else
        Result->AddTextChunk(", (void*)0");
    }
}

static std::string FormatFunctionParameter(ASTContext &Context,
                                           ParmVarDecl *Param) {
  bool ObjCMethodParam = isa<ObjCMethodDecl>(Param->getDeclContext());
  if (Param->getType()->isDependentType() ||
      !Param->getType()->isBlockPointerType()) {
    // The argument for a dependent or non-block parameter is a placeholder 
    // containing that parameter's type.
    std::string Result;
    
    if (Param->getIdentifier() && !ObjCMethodParam)
      Result = Param->getIdentifier()->getName();
    
    Param->getType().getAsStringInternal(Result, 
                                         Context.PrintingPolicy);
    
    if (ObjCMethodParam) {
      Result = "(" + Result;
      Result += ")";
      if (Param->getIdentifier())
        Result += Param->getIdentifier()->getName();
    }
    return Result;
  }
  
  // The argument for a block pointer parameter is a block literal with
  // the appropriate type.
  FunctionProtoTypeLoc *Block = 0;
  TypeLoc TL;
  if (TypeSourceInfo *TSInfo = Param->getTypeSourceInfo()) {
    TL = TSInfo->getTypeLoc().getUnqualifiedLoc();
    while (true) {
      // Look through typedefs.
      if (TypedefTypeLoc *TypedefTL = dyn_cast<TypedefTypeLoc>(&TL)) {
        if (TypeSourceInfo *InnerTSInfo
            = TypedefTL->getTypedefDecl()->getTypeSourceInfo()) {
          TL = InnerTSInfo->getTypeLoc().getUnqualifiedLoc();
          continue;
        }
      }
      
      // Look through qualified types
      if (QualifiedTypeLoc *QualifiedTL = dyn_cast<QualifiedTypeLoc>(&TL)) {
        TL = QualifiedTL->getUnqualifiedLoc();
        continue;
      }
      
      // Try to get the function prototype behind the block pointer type,
      // then we're done.
      if (BlockPointerTypeLoc *BlockPtr
          = dyn_cast<BlockPointerTypeLoc>(&TL)) {
        TL = BlockPtr->getPointeeLoc();
        Block = dyn_cast<FunctionProtoTypeLoc>(&TL);
      }
      break;
    }
  }

  if (!Block) {
    // We were unable to find a FunctionProtoTypeLoc with parameter names
    // for the block; just use the parameter type as a placeholder.
    std::string Result;
    Param->getType().getUnqualifiedType().
                            getAsStringInternal(Result, Context.PrintingPolicy);
    
    if (ObjCMethodParam) {
      Result = "(" + Result;
      Result += ")";
      if (Param->getIdentifier())
        Result += Param->getIdentifier()->getName();
    }
      
    return Result;
  }
  
  // We have the function prototype behind the block pointer type, as it was
  // written in the source.
  std::string Result = "(^)(";
  for (unsigned I = 0, N = Block->getNumArgs(); I != N; ++I) {
    if (I)
      Result += ", ";
    Result += FormatFunctionParameter(Context, Block->getArg(I));
  }
  if (Block->getTypePtr()->isVariadic()) {
    if (Block->getNumArgs() > 0)
      Result += ", ...";
    else
      Result += "...";
  } else if (Block->getNumArgs() == 0 && !Context.getLangOptions().CPlusPlus)
    Result += "void";
             
  Result += ")";
  Block->getTypePtr()->getResultType().getAsStringInternal(Result, 
                                                        Context.PrintingPolicy);
  return Result;
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
    std::string PlaceholderStr = FormatFunctionParameter(Context, Param);
        
    // Add the placeholder string.
    CCStr->AddPlaceholderChunk(PlaceholderStr);
  }
  
  if (const FunctionProtoType *Proto 
        = Function->getType()->getAs<FunctionProtoType>())
    if (Proto->isVariadic()) {
      CCStr->AddPlaceholderChunk(", ...");

      MaybeAddSentinel(Context, Function, CCStr);
    }
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
CodeCompletionResult::CreateCodeCompletionString(Sema &S,
                                               CodeCompletionString *Result) {
  typedef CodeCompletionString::Chunk Chunk;
  
  if (Kind == RK_Pattern)
    return Pattern->Clone(Result);
  
  if (!Result)
    Result = new CodeCompletionString;

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
  
  assert(Kind == RK_Declaration && "Missed a result kind?");
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
        if (Idx < StartParameter || AllParametersAreInformative)
          Result->AddInformativeChunk(Keyword);
        else if (Idx == StartParameter)
          Result->AddTypedTextChunk(Keyword);
        else
          Result->AddTextChunk(Keyword);
      }
      
      // If we're before the starting parameter, skip the placeholder.
      if (Idx < StartParameter)
        continue;

      std::string Arg;
      
      if ((*P)->getType()->isBlockPointerType() && !DeclaringEntity)
        Arg = FormatFunctionParameter(S.Context, *P);
      else {
        (*P)->getType().getAsStringInternal(Arg, S.Context.PrintingPolicy);
        Arg = "(" + Arg + ")";
        if (IdentifierInfo *II = (*P)->getIdentifier())
          Arg += II->getName().str();
      }
      
      if (DeclaringEntity)
        Result->AddTextChunk(Arg);
      else if (AllParametersAreInformative)
        Result->AddInformativeChunk(Arg);
      else
        Result->AddPlaceholderChunk(Arg);
    }

    if (Method->isVariadic()) {
      if (DeclaringEntity)
        Result->AddTextChunk(", ...");
      else if (AllParametersAreInformative)
        Result->AddInformativeChunk(", ...");
      else
        Result->AddPlaceholderChunk(", ...");
      
      MaybeAddSentinel(S.Context, Method, Result);
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

unsigned clang::getMacroUsagePriority(llvm::StringRef MacroName, 
                                      bool PreferredTypeIsPointer) {
  unsigned Priority = CCP_Macro;
  
  // Treat the "nil" and "NULL" macros as null pointer constants.
  if (MacroName.equals("nil") || MacroName.equals("NULL")) {
    Priority = CCP_Constant;
    if (PreferredTypeIsPointer)
      Priority = Priority / CCF_SimilarTypeMatch;
  }
  
  return Priority;
}

static void AddMacroResults(Preprocessor &PP, ResultBuilder &Results,
                            bool TargetTypeIsPointer = false) {
  typedef CodeCompletionResult Result;
  
  Results.EnterNewScope();
  for (Preprocessor::macro_iterator M = PP.macro_begin(), 
                                 MEnd = PP.macro_end();
       M != MEnd; ++M) {
    Results.AddResult(Result(M->first, 
                             getMacroUsagePriority(M->first->getName(),
                                                   TargetTypeIsPointer)));
  }
  Results.ExitScope();
}

static void AddPrettyFunctionResults(const LangOptions &LangOpts, 
                                     ResultBuilder &Results) {
  typedef CodeCompletionResult Result;
  
  Results.EnterNewScope();
  Results.AddResult(Result("__PRETTY_FUNCTION__", CCP_Constant));
  Results.AddResult(Result("__FUNCTION__", CCP_Constant));
  if (LangOpts.C99 || LangOpts.CPlusPlus0x)
    Results.AddResult(Result("__func__", CCP_Constant));
  Results.ExitScope();
}

static void HandleCodeCompleteResults(Sema *S,
                                      CodeCompleteConsumer *CodeCompleter,
                                      CodeCompletionContext Context,
                                      CodeCompletionResult *Results,
                                      unsigned NumResults) {
  std::stable_sort(Results, Results + NumResults);

  if (CodeCompleter)
    CodeCompleter->ProcessCodeCompleteResults(*S, Context, Results, NumResults);
  
  for (unsigned I = 0; I != NumResults; ++I)
    Results[I].Destroy();
}

static enum CodeCompletionContext::Kind mapCodeCompletionContext(Sema &S, 
                                            Sema::ParserCompletionContext PCC) {
  switch (PCC) {
  case Action::PCC_Namespace:
    return CodeCompletionContext::CCC_TopLevel;
      
  case Action::PCC_Class:
    return CodeCompletionContext::CCC_ClassStructUnion;

  case Action::PCC_ObjCInterface:
    return CodeCompletionContext::CCC_ObjCInterface;
      
  case Action::PCC_ObjCImplementation:
    return CodeCompletionContext::CCC_ObjCImplementation;

  case Action::PCC_ObjCInstanceVariableList:
    return CodeCompletionContext::CCC_ObjCIvarList;
      
  case Action::PCC_Template:
  case Action::PCC_MemberTemplate:
  case Action::PCC_RecoveryInFunction:
    return CodeCompletionContext::CCC_Other;
      
  case Action::PCC_Expression:
  case Action::PCC_ForInit:
  case Action::PCC_Condition:
    return CodeCompletionContext::CCC_Expression;
      
  case Action::PCC_Statement:
    return CodeCompletionContext::CCC_Statement;

  case Action::PCC_Type:
    return CodeCompletionContext::CCC_Type;
  }
  
  return CodeCompletionContext::CCC_Other;
}

void Sema::CodeCompleteOrdinaryName(Scope *S, 
                                    ParserCompletionContext CompletionContext) {
  typedef CodeCompletionResult Result;
  ResultBuilder Results(*this);  

  // Determine how to filter results, e.g., so that the names of
  // values (functions, enumerators, function templates, etc.) are
  // only allowed where we can have an expression.
  switch (CompletionContext) {
  case PCC_Namespace:
  case PCC_Class:
  case PCC_ObjCInterface:
  case PCC_ObjCImplementation:
  case PCC_ObjCInstanceVariableList:
  case PCC_Template:
  case PCC_MemberTemplate:
  case PCC_Type:
    Results.setFilter(&ResultBuilder::IsOrdinaryNonValueName);
    break;

  case PCC_Statement:
    // For statements that are expressions, we prefer to call 'void' functions 
    // rather than functions that return a result, since then the result would
    // be ignored.
    Results.setPreferredType(Context.VoidTy);
    // Fall through
      
  case PCC_Expression:
  case PCC_ForInit:
  case PCC_Condition:
    if (WantTypesInContext(CompletionContext, getLangOptions()))
      Results.setFilter(&ResultBuilder::IsOrdinaryName);
    else
      Results.setFilter(&ResultBuilder::IsOrdinaryNonTypeName);
    break;
      
  case PCC_RecoveryInFunction:
    // Unfiltered
    break;
  }

  CodeCompletionDeclConsumer Consumer(Results, CurContext);
  LookupVisibleDecls(S, LookupOrdinaryName, Consumer,
                     CodeCompleter->includeGlobals());

  Results.EnterNewScope();
  AddOrdinaryNameResults(CompletionContext, S, *this, Results);
  Results.ExitScope();

  switch (CompletionContext) {
  case PCC_Expression:
  case PCC_Statement:
  case PCC_RecoveryInFunction:
    if (S->getFnParent())
      AddPrettyFunctionResults(PP.getLangOptions(), Results);        
    break;
    
  case PCC_Namespace:
  case PCC_Class:
  case PCC_ObjCInterface:
  case PCC_ObjCImplementation:
  case PCC_ObjCInstanceVariableList:
  case PCC_Template:
  case PCC_MemberTemplate:
  case PCC_ForInit:
  case PCC_Condition:
  case PCC_Type:
    break;
  }
  
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, Results);
  
  HandleCodeCompleteResults(this, CodeCompleter,
                            mapCodeCompletionContext(*this, CompletionContext),
                            Results.data(),Results.size());
}

void Sema::CodeCompleteDeclarator(Scope *S,
                                  bool AllowNonIdentifiers,
                                  bool AllowNestedNameSpecifiers) {
  typedef CodeCompletionResult Result;
  ResultBuilder Results(*this);    
  Results.EnterNewScope();
  
  // Type qualifiers can come after names.
  Results.AddResult(Result("const"));
  Results.AddResult(Result("volatile"));
  if (getLangOptions().C99)
    Results.AddResult(Result("restrict"));

  if (getLangOptions().CPlusPlus) {
    if (AllowNonIdentifiers) {
      Results.AddResult(Result("operator")); 
    }
    
    // Add nested-name-specifiers.
    if (AllowNestedNameSpecifiers) {
      Results.allowNestedNameSpecifiers();
      CodeCompletionDeclConsumer Consumer(Results, CurContext);
      LookupVisibleDecls(S, LookupNestedNameSpecifierName, Consumer,
                         CodeCompleter->includeGlobals());
    }
  }
  Results.ExitScope();

  // Note that we intentionally suppress macro results here, since we do not
  // encourage using macros to produce the names of entities.

  HandleCodeCompleteResults(this, CodeCompleter,
                        AllowNestedNameSpecifiers
                          ? CodeCompletionContext::CCC_PotentiallyQualifiedName
                          : CodeCompletionContext::CCC_Name,
                            Results.data(), Results.size());
}

struct Sema::CodeCompleteExpressionData {
  CodeCompleteExpressionData(QualType PreferredType = QualType()) 
    : PreferredType(PreferredType), IntegralConstantExpression(false),
      ObjCCollection(false) { }
  
  QualType PreferredType;
  bool IntegralConstantExpression;
  bool ObjCCollection;
  llvm::SmallVector<Decl *, 4> IgnoreDecls;
};

/// \brief Perform code-completion in an expression context when we know what
/// type we're looking for.
///
/// \param IntegralConstantExpression Only permit integral constant 
/// expressions.
void Sema::CodeCompleteExpression(Scope *S, 
                                  const CodeCompleteExpressionData &Data) {
  typedef CodeCompletionResult Result;
  ResultBuilder Results(*this);
  
  if (Data.ObjCCollection)
    Results.setFilter(&ResultBuilder::IsObjCCollection);
  else if (Data.IntegralConstantExpression)
    Results.setFilter(&ResultBuilder::IsIntegralConstantValue);
  else if (WantTypesInContext(PCC_Expression, getLangOptions()))
    Results.setFilter(&ResultBuilder::IsOrdinaryName);
  else
    Results.setFilter(&ResultBuilder::IsOrdinaryNonTypeName);

  if (!Data.PreferredType.isNull())
    Results.setPreferredType(Data.PreferredType.getNonReferenceType());
  
  // Ignore any declarations that we were told that we don't care about.
  for (unsigned I = 0, N = Data.IgnoreDecls.size(); I != N; ++I)
    Results.Ignore(Data.IgnoreDecls[I]);
  
  CodeCompletionDeclConsumer Consumer(Results, CurContext);
  LookupVisibleDecls(S, LookupOrdinaryName, Consumer,
                     CodeCompleter->includeGlobals());
  
  Results.EnterNewScope();
  AddOrdinaryNameResults(PCC_Expression, S, *this, Results);
  Results.ExitScope();
  
  bool PreferredTypeIsPointer = false;
  if (!Data.PreferredType.isNull())
    PreferredTypeIsPointer = Data.PreferredType->isAnyPointerType()
      || Data.PreferredType->isMemberPointerType() 
      || Data.PreferredType->isBlockPointerType();
  
  if (S->getFnParent() && 
      !Data.ObjCCollection && 
      !Data.IntegralConstantExpression)
    AddPrettyFunctionResults(PP.getLangOptions(), Results);        

  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, Results, PreferredTypeIsPointer);
  HandleCodeCompleteResults(this, CodeCompleter, 
                CodeCompletionContext(CodeCompletionContext::CCC_Expression, 
                                      Data.PreferredType),
                            Results.data(),Results.size());
}


static void AddObjCProperties(ObjCContainerDecl *Container, 
                              bool AllowCategories,
                              DeclContext *CurContext,
                              ResultBuilder &Results) {
  typedef CodeCompletionResult Result;

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
  
  typedef CodeCompletionResult Result;
  
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
    Results.allowNestedNameSpecifiers();
    CodeCompletionDeclConsumer Consumer(Results, CurContext);
    LookupVisibleDecls(Record->getDecl(), LookupMemberName, Consumer,
                       CodeCompleter->includeGlobals());

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
          Results.AddResult(Result("template"));
      }
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
             (!IsArrow && BaseType->isObjCObjectType())) {
    // Objective-C instance variable access.
    ObjCInterfaceDecl *Class = 0;
    if (const ObjCObjectPointerType *ObjCPtr
                                    = BaseType->getAs<ObjCObjectPointerType>())
      Class = ObjCPtr->getInterfaceDecl();
    else
      Class = BaseType->getAs<ObjCObjectType>()->getInterface();
    
    // Add all ivars from this class and its superclasses.
    if (Class) {
      CodeCompletionDeclConsumer Consumer(Results, CurContext);
      Results.setFilter(&ResultBuilder::IsObjCIvar);
      LookupVisibleDecls(Class, LookupMemberName, Consumer,
                         CodeCompleter->includeGlobals());
    }
  }
  
  // FIXME: How do we cope with isa?
  
  Results.ExitScope();

  // Hand off the results found for code completion.
  HandleCodeCompleteResults(this, CodeCompleter, 
            CodeCompletionContext(CodeCompletionContext::CCC_MemberAccess,
                                              BaseType),
                            Results.data(),Results.size());
}

void Sema::CodeCompleteTag(Scope *S, unsigned TagSpec) {
  if (!CodeCompleter)
    return;
  
  typedef CodeCompletionResult Result;
  ResultBuilder::LookupFilter Filter = 0;
  enum CodeCompletionContext::Kind ContextKind
    = CodeCompletionContext::CCC_Other;
  switch ((DeclSpec::TST)TagSpec) {
  case DeclSpec::TST_enum:
    Filter = &ResultBuilder::IsEnum;
    ContextKind = CodeCompletionContext::CCC_EnumTag;
    break;
    
  case DeclSpec::TST_union:
    Filter = &ResultBuilder::IsUnion;
    ContextKind = CodeCompletionContext::CCC_UnionTag;
    break;
    
  case DeclSpec::TST_struct:
  case DeclSpec::TST_class:
    Filter = &ResultBuilder::IsClassOrStruct;
    ContextKind = CodeCompletionContext::CCC_ClassOrStructTag;
    break;
    
  default:
    assert(false && "Unknown type specifier kind in CodeCompleteTag");
    return;
  }
  
  ResultBuilder Results(*this);
  CodeCompletionDeclConsumer Consumer(Results, CurContext);

  // First pass: look for tags.
  Results.setFilter(Filter);
  LookupVisibleDecls(S, LookupTagName, Consumer,
                     CodeCompleter->includeGlobals());

  if (CodeCompleter->includeGlobals()) {
    // Second pass: look for nested name specifiers.
    Results.setFilter(&ResultBuilder::IsNestedNameSpecifier);
    LookupVisibleDecls(S, LookupNestedNameSpecifierName, Consumer);
  }
  
  HandleCodeCompleteResults(this, CodeCompleter, ContextKind,
                            Results.data(),Results.size());
}

void Sema::CodeCompleteCase(Scope *S) {
  if (getCurFunction()->SwitchStack.empty() || !CodeCompleter)
    return;
  
  SwitchStmt *Switch = getCurFunction()->SwitchStack.back();
  if (!Switch->getCond()->getType()->isEnumeralType()) {
    CodeCompleteExpressionData Data(Switch->getCond()->getType());
    Data.IntegralConstantExpression = true;
    CodeCompleteExpression(S, Data);
    return;
  }
  
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
    
    Results.AddResult(CodeCompletionResult(*E, Qualifier),
                      CurContext, 0, false);
  }
  Results.ExitScope();

  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, Results);
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Expression,
                            Results.data(),Results.size());
}

namespace {
  struct IsBetterOverloadCandidate {
    Sema &S;
    SourceLocation Loc;
    
  public:
    explicit IsBetterOverloadCandidate(Sema &S, SourceLocation Loc)
      : S(S), Loc(Loc) { }
    
    bool 
    operator()(const OverloadCandidate &X, const OverloadCandidate &Y) const {
      return isBetterOverloadCandidate(S, X, Y, Loc);
    }
  };
}

static bool anyNullArguments(Expr **Args, unsigned NumArgs) {
  if (NumArgs && !Args)
    return true;
  
  for (unsigned I = 0; I != NumArgs; ++I)
    if (!Args[I])
      return true;
  
  return false;
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
  if (!Fn || Fn->isTypeDependent() || anyNullArguments(Args, NumArgs) ||
      Expr::hasAnyTypeDependentArguments(Args, NumArgs)) {
    CodeCompleteOrdinaryName(S, PCC_Expression);
    return;
  }

  // Build an overload candidate set based on the functions we find.
  SourceLocation Loc = Fn->getExprLoc();
  OverloadCandidateSet CandidateSet(Loc);

  // FIXME: What if we're calling something that isn't a function declaration?
  // FIXME: What if we're calling a pseudo-destructor?
  // FIXME: What if we're calling a member function?
  
  typedef CodeCompleteConsumer::OverloadCandidate ResultCandidate;
  llvm::SmallVector<ResultCandidate, 8> Results;

  Expr *NakedFn = Fn->IgnoreParenCasts();
  if (UnresolvedLookupExpr *ULE = dyn_cast<UnresolvedLookupExpr>(NakedFn))
    AddOverloadedCallCandidates(ULE, Args, NumArgs, CandidateSet,
                                /*PartialOverloading=*/ true);
  else if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(NakedFn)) {
    FunctionDecl *FDecl = dyn_cast<FunctionDecl>(DRE->getDecl());
    if (FDecl) {
      if (!getLangOptions().CPlusPlus || 
          !FDecl->getType()->getAs<FunctionProtoType>())
        Results.push_back(ResultCandidate(FDecl));
      else
        // FIXME: access?
        AddOverloadCandidate(FDecl, DeclAccessPair::make(FDecl, AS_none),
                             Args, NumArgs, CandidateSet,
                             false, /*PartialOverloading*/true);
    }
  }
  
  QualType ParamType;
  
  if (!CandidateSet.empty()) {
    // Sort the overload candidate set by placing the best overloads first.
    std::stable_sort(CandidateSet.begin(), CandidateSet.end(),
                     IsBetterOverloadCandidate(*this, Loc));
  
    // Add the remaining viable overload candidates as code-completion reslults.
    for (OverloadCandidateSet::iterator Cand = CandidateSet.begin(),
                                     CandEnd = CandidateSet.end();
         Cand != CandEnd; ++Cand) {
      if (Cand->Viable)
        Results.push_back(ResultCandidate(Cand->Function));
    }

    // From the viable candidates, try to determine the type of this parameter.
    for (unsigned I = 0, N = Results.size(); I != N; ++I) {
      if (const FunctionType *FType = Results[I].getFunctionType())
        if (const FunctionProtoType *Proto = dyn_cast<FunctionProtoType>(FType))
          if (NumArgs < Proto->getNumArgs()) {
            if (ParamType.isNull())
              ParamType = Proto->getArgType(NumArgs);
            else if (!Context.hasSameUnqualifiedType(
                                            ParamType.getNonReferenceType(),
                           Proto->getArgType(NumArgs).getNonReferenceType())) {
              ParamType = QualType();
              break;
            }
          }
    }
  } else {
    // Try to determine the parameter type from the type of the expression
    // being called.
    QualType FunctionType = Fn->getType();
    if (const PointerType *Ptr = FunctionType->getAs<PointerType>())
      FunctionType = Ptr->getPointeeType();
    else if (const BlockPointerType *BlockPtr
                                    = FunctionType->getAs<BlockPointerType>())
      FunctionType = BlockPtr->getPointeeType();
    else if (const MemberPointerType *MemPtr
                                    = FunctionType->getAs<MemberPointerType>())
      FunctionType = MemPtr->getPointeeType();
    
    if (const FunctionProtoType *Proto
                                  = FunctionType->getAs<FunctionProtoType>()) {
      if (NumArgs < Proto->getNumArgs())
        ParamType = Proto->getArgType(NumArgs);
    }
  }

  if (ParamType.isNull())
    CodeCompleteOrdinaryName(S, PCC_Expression);
  else
    CodeCompleteExpression(S, ParamType);
  
  if (!Results.empty())
    CodeCompleter->ProcessOverloadCandidates(*this, NumArgs, Results.data(), 
                                             Results.size());
}

void Sema::CodeCompleteInitializer(Scope *S, Decl *D) {
  ValueDecl *VD = dyn_cast_or_null<ValueDecl>(D);
  if (!VD) {
    CodeCompleteOrdinaryName(S, PCC_Expression);
    return;
  }
  
  CodeCompleteExpression(S, VD->getType());
}

void Sema::CodeCompleteReturn(Scope *S) {
  QualType ResultType;
  if (isa<BlockDecl>(CurContext)) {
    if (BlockScopeInfo *BSI = getCurBlock())
      ResultType = BSI->ReturnType;
  } else if (FunctionDecl *Function = dyn_cast<FunctionDecl>(CurContext))
    ResultType = Function->getResultType();
  else if (ObjCMethodDecl *Method = dyn_cast<ObjCMethodDecl>(CurContext))
    ResultType = Method->getResultType();
  
  if (ResultType.isNull())
    CodeCompleteOrdinaryName(S, PCC_Expression);
  else
    CodeCompleteExpression(S, ResultType);
}

void Sema::CodeCompleteAssignmentRHS(Scope *S, ExprTy *LHS) {
  if (LHS)
    CodeCompleteExpression(S, static_cast<Expr *>(LHS)->getType());
  else
    CodeCompleteOrdinaryName(S, PCC_Expression);
}

void Sema::CodeCompleteQualifiedId(Scope *S, CXXScopeSpec &SS,
                                   bool EnteringContext) {
  if (!SS.getScopeRep() || !CodeCompleter)
    return;
  
  DeclContext *Ctx = computeDeclContext(SS, EnteringContext);
  if (!Ctx)
    return;

  // Try to instantiate any non-dependent declaration contexts before
  // we look in them.
  if (!isDependentScopeSpecifier(SS) && RequireCompleteDeclContext(SS, Ctx))
    return;

  ResultBuilder Results(*this);
  CodeCompletionDeclConsumer Consumer(Results, CurContext);
  LookupVisibleDecls(Ctx, LookupOrdinaryName, Consumer);
  
  // The "template" keyword can follow "::" in the grammar, but only
  // put it into the grammar if the nested-name-specifier is dependent.
  NestedNameSpecifier *NNS = (NestedNameSpecifier *)SS.getScopeRep();
  if (!Results.empty() && NNS->isDependent())
    Results.AddResult("template");
  
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
}

void Sema::CodeCompleteUsing(Scope *S) {
  if (!CodeCompleter)
    return;
  
  ResultBuilder Results(*this, &ResultBuilder::IsNestedNameSpecifier);
  Results.EnterNewScope();
  
  // If we aren't in class scope, we could see the "namespace" keyword.
  if (!S->isClassScope())
    Results.AddResult(CodeCompletionResult("namespace"));
  
  // After "using", we can see anything that would start a 
  // nested-name-specifier.
  CodeCompletionDeclConsumer Consumer(Results, CurContext);
  LookupVisibleDecls(S, LookupOrdinaryName, Consumer,
                     CodeCompleter->includeGlobals());
  Results.ExitScope();
  
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
}

void Sema::CodeCompleteUsingDirective(Scope *S) {
  if (!CodeCompleter)
    return;
  
  // After "using namespace", we expect to see a namespace name or namespace
  // alias.
  ResultBuilder Results(*this, &ResultBuilder::IsNamespaceOrAlias);
  Results.EnterNewScope();
  CodeCompletionDeclConsumer Consumer(Results, CurContext);
  LookupVisibleDecls(S, LookupOrdinaryName, Consumer,
                     CodeCompleter->includeGlobals());
  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Namespace,
                            Results.data(),Results.size());
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
      Results.AddResult(CodeCompletionResult(NS->second, 0),
                        CurContext, 0, false);
    Results.ExitScope();
  }
  
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
}

void Sema::CodeCompleteNamespaceAliasDecl(Scope *S)  {
  if (!CodeCompleter)
    return;
  
  // After "namespace", we expect to see a namespace or alias.
  ResultBuilder Results(*this, &ResultBuilder::IsNamespaceOrAlias);
  CodeCompletionDeclConsumer Consumer(Results, CurContext);
  LookupVisibleDecls(S, LookupOrdinaryName, Consumer,
                     CodeCompleter->includeGlobals());
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Namespace,
                            Results.data(),Results.size());
}

void Sema::CodeCompleteOperatorName(Scope *S) {
  if (!CodeCompleter)
    return;

  typedef CodeCompletionResult Result;
  ResultBuilder Results(*this, &ResultBuilder::IsType);
  Results.EnterNewScope();
  
  // Add the names of overloadable operators.
#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly)      \
  if (std::strcmp(Spelling, "?"))                                                  \
    Results.AddResult(Result(Spelling));
#include "clang/Basic/OperatorKinds.def"
  
  // Add any type names visible from the current scope
  Results.allowNestedNameSpecifiers();
  CodeCompletionDeclConsumer Consumer(Results, CurContext);
  LookupVisibleDecls(S, LookupOrdinaryName, Consumer,
                     CodeCompleter->includeGlobals());
  
  // Add any type specifiers
  AddTypeSpecifierResults(getLangOptions(), Results);
  Results.ExitScope();
  
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Type,
                            Results.data(),Results.size());
}

// Macro that expands to @Keyword or Keyword, depending on whether NeedAt is
// true or false.
#define OBJC_AT_KEYWORD_NAME(NeedAt,Keyword) NeedAt? "@" #Keyword : #Keyword
static void AddObjCImplementationResults(const LangOptions &LangOpts,
                                         ResultBuilder &Results,
                                         bool NeedAt) {
  typedef CodeCompletionResult Result;
  // Since we have an implementation, we can end it.
  Results.AddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,end)));
  
  CodeCompletionString *Pattern = 0;
  if (LangOpts.ObjC2) {
    // @dynamic
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,dynamic));
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddPlaceholderChunk("property");
    Results.AddResult(Result(Pattern));
    
    // @synthesize
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,synthesize));
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddPlaceholderChunk("property");
    Results.AddResult(Result(Pattern));
  }  
}

static void AddObjCInterfaceResults(const LangOptions &LangOpts,
                                    ResultBuilder &Results,
                                    bool NeedAt) {
  typedef CodeCompletionResult Result;
  
  // Since we have an interface or protocol, we can end it.
  Results.AddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,end)));
  
  if (LangOpts.ObjC2) {
    // @property
    Results.AddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,property)));
  
    // @required
    Results.AddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,required)));
  
    // @optional
    Results.AddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,optional)));
  }
}

static void AddObjCTopLevelResults(ResultBuilder &Results, bool NeedAt) {
  typedef CodeCompletionResult Result;
  CodeCompletionString *Pattern = 0;
  
  // @class name ;
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,class));
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("name");
  Results.AddResult(Result(Pattern));
  
  if (Results.includeCodePatterns()) {
    // @interface name 
    // FIXME: Could introduce the whole pattern, including superclasses and 
    // such.
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,interface));
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddPlaceholderChunk("class");
    Results.AddResult(Result(Pattern));
  
    // @protocol name
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,protocol));
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddPlaceholderChunk("protocol");
    Results.AddResult(Result(Pattern));
    
    // @implementation name
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,implementation));
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddPlaceholderChunk("class");
    Results.AddResult(Result(Pattern));
  }
  
  // @compatibility_alias name
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,compatibility_alias));
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("alias");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("class");
  Results.AddResult(Result(Pattern));
}

void Sema::CodeCompleteObjCAtDirective(Scope *S, Decl *ObjCImpDecl,
                                       bool InInterface) {
  typedef CodeCompletionResult Result;
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  if (ObjCImpDecl)
    AddObjCImplementationResults(getLangOptions(), Results, false);
  else if (InInterface)
    AddObjCInterfaceResults(getLangOptions(), Results, false);
  else
    AddObjCTopLevelResults(Results, false);
  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
}

static void AddObjCExpressionResults(ResultBuilder &Results, bool NeedAt) {
  typedef CodeCompletionResult Result;
  CodeCompletionString *Pattern = 0;

  // @encode ( type-name )
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,encode));
  Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
  Pattern->AddPlaceholderChunk("type-name");
  Pattern->AddChunk(CodeCompletionString::CK_RightParen);
  Results.AddResult(Result(Pattern));
  
  // @protocol ( protocol-name )
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,protocol));
  Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
  Pattern->AddPlaceholderChunk("protocol-name");
  Pattern->AddChunk(CodeCompletionString::CK_RightParen);
  Results.AddResult(Result(Pattern));

  // @selector ( selector )
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,selector));
  Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
  Pattern->AddPlaceholderChunk("selector");
  Pattern->AddChunk(CodeCompletionString::CK_RightParen);
  Results.AddResult(Result(Pattern));
}

static void AddObjCStatementResults(ResultBuilder &Results, bool NeedAt) {
  typedef CodeCompletionResult Result;
  CodeCompletionString *Pattern = 0;
  
  if (Results.includeCodePatterns()) {
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
    Results.AddResult(Result(Pattern));
  }
  
  // @throw
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk(OBJC_AT_KEYWORD_NAME(NeedAt,throw));
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("expression");
  Results.AddResult(Result(Pattern));
  
  if (Results.includeCodePatterns()) {
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
    Results.AddResult(Result(Pattern));
  }
}

static void AddObjCVisibilityResults(const LangOptions &LangOpts,
                                     ResultBuilder &Results,
                                     bool NeedAt) {
  typedef CodeCompletionResult Result;
  Results.AddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,private)));
  Results.AddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,protected)));
  Results.AddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,public)));
  if (LangOpts.ObjC2)
    Results.AddResult(Result(OBJC_AT_KEYWORD_NAME(NeedAt,package)));    
}

void Sema::CodeCompleteObjCAtVisibility(Scope *S) {
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  AddObjCVisibilityResults(getLangOptions(), Results, false);
  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
}

void Sema::CodeCompleteObjCAtStatement(Scope *S) {
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  AddObjCStatementResults(Results, false);
  AddObjCExpressionResults(Results, false);
  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
}

void Sema::CodeCompleteObjCAtExpression(Scope *S) {
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  AddObjCExpressionResults(Results, false);
  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
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
  
  typedef CodeCompletionResult Result;
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  if (!ObjCPropertyFlagConflicts(Attributes, ObjCDeclSpec::DQ_PR_readonly))
    Results.AddResult(CodeCompletionResult("readonly"));
  if (!ObjCPropertyFlagConflicts(Attributes, ObjCDeclSpec::DQ_PR_assign))
    Results.AddResult(CodeCompletionResult("assign"));
  if (!ObjCPropertyFlagConflicts(Attributes, ObjCDeclSpec::DQ_PR_readwrite))
    Results.AddResult(CodeCompletionResult("readwrite"));
  if (!ObjCPropertyFlagConflicts(Attributes, ObjCDeclSpec::DQ_PR_retain))
    Results.AddResult(CodeCompletionResult("retain"));
  if (!ObjCPropertyFlagConflicts(Attributes, ObjCDeclSpec::DQ_PR_copy))
    Results.AddResult(CodeCompletionResult("copy"));
  if (!ObjCPropertyFlagConflicts(Attributes, ObjCDeclSpec::DQ_PR_nonatomic))
    Results.AddResult(CodeCompletionResult("nonatomic"));
  if (!ObjCPropertyFlagConflicts(Attributes, ObjCDeclSpec::DQ_PR_setter)) {
    CodeCompletionString *Setter = new CodeCompletionString;
    Setter->AddTypedTextChunk("setter");
    Setter->AddTextChunk(" = ");
    Setter->AddPlaceholderChunk("method");
    Results.AddResult(CodeCompletionResult(Setter));
  }
  if (!ObjCPropertyFlagConflicts(Attributes, ObjCDeclSpec::DQ_PR_getter)) {
    CodeCompletionString *Getter = new CodeCompletionString;
    Getter->AddTypedTextChunk("getter");
    Getter->AddTextChunk(" = ");
    Getter->AddPlaceholderChunk("method");
    Results.AddResult(CodeCompletionResult(Getter));
  }
  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
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
                           ResultBuilder &Results,
                           bool InOriginalClass = true) {
  typedef CodeCompletionResult Result;
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
      if (!InOriginalClass)
        R.Priority += CCD_InBaseClass;
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
                   CurContext, Results, false);
  
  // Add methods in categories.
  for (ObjCCategoryDecl *CatDecl = IFace->getCategoryList(); CatDecl;
       CatDecl = CatDecl->getNextClassCategory()) {
    AddObjCMethods(CatDecl, WantInstanceMethods, WantKind, SelIdents, 
                   NumSelIdents, CurContext, Results, InOriginalClass);
    
    // Add a categories protocol methods.
    const ObjCList<ObjCProtocolDecl> &Protocols 
      = CatDecl->getReferencedProtocols();
    for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
                                              E = Protocols.end();
         I != E; ++I)
      AddObjCMethods(*I, WantInstanceMethods, WantKind, SelIdents, 
                     NumSelIdents, CurContext, Results, false);
    
    // Add methods in category implementations.
    if (ObjCCategoryImplDecl *Impl = CatDecl->getImplementation())
      AddObjCMethods(Impl, WantInstanceMethods, WantKind, SelIdents, 
                     NumSelIdents, CurContext, Results, InOriginalClass);
  }
  
  // Add methods in superclass.
  if (IFace->getSuperClass())
    AddObjCMethods(IFace->getSuperClass(), WantInstanceMethods, WantKind, 
                   SelIdents, NumSelIdents, CurContext, Results, false);

  // Add methods in our implementation, if any.
  if (ObjCImplementationDecl *Impl = IFace->getImplementation())
    AddObjCMethods(Impl, WantInstanceMethods, WantKind, SelIdents,
                   NumSelIdents, CurContext, Results, InOriginalClass);
}


void Sema::CodeCompleteObjCPropertyGetter(Scope *S, Decl *ClassDecl,
                                          Decl **Methods,
                                          unsigned NumMethods) {
  typedef CodeCompletionResult Result;

  // Try to find the interface where getters might live.
  ObjCInterfaceDecl *Class = dyn_cast_or_null<ObjCInterfaceDecl>(ClassDecl);
  if (!Class) {
    if (ObjCCategoryDecl *Category
          = dyn_cast_or_null<ObjCCategoryDecl>(ClassDecl))
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
            = dyn_cast_or_null<ObjCMethodDecl>(Methods[I]))
      if (Method->isInstanceMethod() &&
          isAcceptableObjCMethod(Method, MK_ZeroArgSelector, 0, 0)) {
        Result R = Result(Method, 0);
        R.AllParametersAreInformative = true;
        Results.MaybeAddResult(R, CurContext);
      }
  }

  AddObjCMethods(Class, true, MK_ZeroArgSelector, 0, 0, CurContext, Results);
  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter,
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
}

void Sema::CodeCompleteObjCPropertySetter(Scope *S, Decl *ObjCImplDecl,
                                          Decl **Methods,
                                          unsigned NumMethods) {
  typedef CodeCompletionResult Result;

  // Try to find the interface where setters might live.
  ObjCInterfaceDecl *Class
    = dyn_cast_or_null<ObjCInterfaceDecl>(ObjCImplDecl);
  if (!Class) {
    if (ObjCCategoryDecl *Category
          = dyn_cast_or_null<ObjCCategoryDecl>(ObjCImplDecl))
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
            = dyn_cast_or_null<ObjCMethodDecl>(Methods[I]))
      if (Method->isInstanceMethod() &&
          isAcceptableObjCMethod(Method, MK_OneArgSelector, 0, 0)) {
        Result R = Result(Method, 0);
        R.AllParametersAreInformative = true;
        Results.MaybeAddResult(R, CurContext);
      }
  }

  AddObjCMethods(Class, true, MK_OneArgSelector, 0, 0, CurContext, Results);

  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter,
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
}

void Sema::CodeCompleteObjCPassingType(Scope *S, ObjCDeclSpec &DS) {
  typedef CodeCompletionResult Result;
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  
  // Add context-sensitive, Objective-C parameter-passing keywords.
  bool AddedInOut = false;
  if ((DS.getObjCDeclQualifier() & 
       (ObjCDeclSpec::DQ_In | ObjCDeclSpec::DQ_Inout)) == 0) {
    Results.AddResult("in");
    Results.AddResult("inout");
    AddedInOut = true;
  }
  if ((DS.getObjCDeclQualifier() & 
       (ObjCDeclSpec::DQ_Out | ObjCDeclSpec::DQ_Inout)) == 0) {
    Results.AddResult("out");
    if (!AddedInOut)
      Results.AddResult("inout");
  }
  if ((DS.getObjCDeclQualifier() & 
       (ObjCDeclSpec::DQ_Bycopy | ObjCDeclSpec::DQ_Byref |
        ObjCDeclSpec::DQ_Oneway)) == 0) {
     Results.AddResult("bycopy");
     Results.AddResult("byref");
     Results.AddResult("oneway");
  }
  
  // Add various builtin type names and specifiers.
  AddOrdinaryNameResults(PCC_Type, S, *this, Results);
  Results.ExitScope();
  
  // Add the various type names
  Results.setFilter(&ResultBuilder::IsOrdinaryNonValueName);
  CodeCompletionDeclConsumer Consumer(Results, CurContext);
  LookupVisibleDecls(S, LookupOrdinaryName, Consumer,
                     CodeCompleter->includeGlobals());
  
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, Results);

  HandleCodeCompleteResults(this, CodeCompleter,
                            CodeCompletionContext::CCC_Type,
                            Results.data(), Results.size());
}

/// \brief When we have an expression with type "id", we may assume
/// that it has some more-specific class type based on knowledge of
/// common uses of Objective-C. This routine returns that class type,
/// or NULL if no better result could be determined.
static ObjCInterfaceDecl *GetAssumedMessageSendExprType(Expr *E) {
  ObjCMessageExpr *Msg = dyn_cast<ObjCMessageExpr>(E);
  if (!Msg)
    return 0;

  Selector Sel = Msg->getSelector();
  if (Sel.isNull())
    return 0;

  IdentifierInfo *Id = Sel.getIdentifierInfoForSlot(0);
  if (!Id)
    return 0;

  ObjCMethodDecl *Method = Msg->getMethodDecl();
  if (!Method)
    return 0;

  // Determine the class that we're sending the message to.
  ObjCInterfaceDecl *IFace = 0;
  switch (Msg->getReceiverKind()) {
  case ObjCMessageExpr::Class:
    if (const ObjCObjectType *ObjType
                           = Msg->getClassReceiver()->getAs<ObjCObjectType>())
      IFace = ObjType->getInterface();
    break;

  case ObjCMessageExpr::Instance: {
    QualType T = Msg->getInstanceReceiver()->getType();
    if (const ObjCObjectPointerType *Ptr = T->getAs<ObjCObjectPointerType>())
      IFace = Ptr->getInterfaceDecl();
    break;
  }

  case ObjCMessageExpr::SuperInstance:
  case ObjCMessageExpr::SuperClass:
    break;
  }

  if (!IFace)
    return 0;

  ObjCInterfaceDecl *Super = IFace->getSuperClass();
  if (Method->isInstanceMethod())
    return llvm::StringSwitch<ObjCInterfaceDecl *>(Id->getName())
      .Case("retain", IFace)
      .Case("autorelease", IFace)
      .Case("copy", IFace)
      .Case("copyWithZone", IFace)
      .Case("mutableCopy", IFace)
      .Case("mutableCopyWithZone", IFace)
      .Case("awakeFromCoder", IFace)
      .Case("replacementObjectFromCoder", IFace)
      .Case("class", IFace)
      .Case("classForCoder", IFace)
      .Case("superclass", Super)
      .Default(0);

  return llvm::StringSwitch<ObjCInterfaceDecl *>(Id->getName())
    .Case("new", IFace)
    .Case("alloc", IFace)
    .Case("allocWithZone", IFace)
    .Case("class", IFace)
    .Case("superclass", Super)
    .Default(0);
}

void Sema::CodeCompleteObjCMessageReceiver(Scope *S) {
  typedef CodeCompletionResult Result;
  ResultBuilder Results(*this);
  
  // Find anything that looks like it could be a message receiver.
  Results.setFilter(&ResultBuilder::IsObjCMessageReceiver);
  CodeCompletionDeclConsumer Consumer(Results, CurContext);
  Results.EnterNewScope();
  LookupVisibleDecls(S, LookupOrdinaryName, Consumer,
                     CodeCompleter->includeGlobals());
  
  // If we are in an Objective-C method inside a class that has a superclass,
  // add "super" as an option.
  if (ObjCMethodDecl *Method = getCurMethodDecl())
    if (ObjCInterfaceDecl *Iface = Method->getClassInterface())
      if (Iface->getSuperClass())
        Results.AddResult(Result("super"));
  
  Results.ExitScope();
  
  if (CodeCompleter->includeMacros())
    AddMacroResults(PP, Results);
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_ObjCMessageReceiver,
                            Results.data(), Results.size());
  
}

void Sema::CodeCompleteObjCSuperMessage(Scope *S, SourceLocation SuperLoc,
                                        IdentifierInfo **SelIdents,
                                        unsigned NumSelIdents) {
  ObjCInterfaceDecl *CDecl = 0;
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
      ExprResult Super
        = Owned(new (Context) ObjCSuperExpr(SuperLoc, SuperTy));
      return CodeCompleteObjCInstanceMessage(S, (Expr *)Super.get(),
                                             SelIdents, NumSelIdents);
    }

    // Fall through to send to the superclass in CDecl.
  } else {
    // "super" may be the name of a type or variable. Figure out which
    // it is.
    IdentifierInfo *Super = &Context.Idents.get("super");
    NamedDecl *ND = LookupSingleName(S, Super, SuperLoc, 
                                     LookupOrdinaryName);
    if ((CDecl = dyn_cast_or_null<ObjCInterfaceDecl>(ND))) {
      // "super" names an interface. Use it.
    } else if (TypeDecl *TD = dyn_cast_or_null<TypeDecl>(ND)) {
      if (const ObjCObjectType *Iface
            = Context.getTypeDeclType(TD)->getAs<ObjCObjectType>())
        CDecl = Iface->getInterface();
    } else if (ND && isa<UnresolvedUsingTypenameDecl>(ND)) {
      // "super" names an unresolved type; we can't be more specific.
    } else {
      // Assume that "super" names some kind of value and parse that way.
      CXXScopeSpec SS;
      UnqualifiedId id;
      id.setIdentifier(Super, SuperLoc);
      ExprResult SuperExpr = ActOnIdExpression(S, SS, id, false, false);
      return CodeCompleteObjCInstanceMessage(S, (Expr *)SuperExpr.get(),
                                             SelIdents, NumSelIdents);
    }

    // Fall through
  }

  ParsedType Receiver;
  if (CDecl)
    Receiver = ParsedType::make(Context.getObjCInterfaceType(CDecl));
  return CodeCompleteObjCClassMessage(S, Receiver, SelIdents, 
                                      NumSelIdents);
}

void Sema::CodeCompleteObjCClassMessage(Scope *S, ParsedType Receiver,
                                        IdentifierInfo **SelIdents,
                                        unsigned NumSelIdents) {
  typedef CodeCompletionResult Result;
  ObjCInterfaceDecl *CDecl = 0;

  // If the given name refers to an interface type, retrieve the
  // corresponding declaration.
  if (Receiver) {
    QualType T = GetTypeFromParser(Receiver, 0);
    if (!T.isNull()) 
      if (const ObjCObjectType *Interface = T->getAs<ObjCObjectType>())
        CDecl = Interface->getInterface();
  }

  // Add all of the factory methods in this Objective-C class, its protocols,
  // superclasses, categories, implementation, etc.
  ResultBuilder Results(*this);
  Results.EnterNewScope();

  if (CDecl) 
    AddObjCMethods(CDecl, false, MK_Any, SelIdents, NumSelIdents, CurContext, 
                   Results);  
  else {
    // We're messaging "id" as a type; provide all class/factory methods.

    // If we have an external source, load the entire class method
    // pool from the AST file.
    if (ExternalSource) {
      for (uint32_t I = 0, N = ExternalSource->GetNumExternalSelectors();
           I != N; ++I) {
        Selector Sel = ExternalSource->GetExternalSelector(I);
        if (Sel.isNull() || MethodPool.count(Sel))
          continue;

        ReadMethodPool(Sel);
      }
    }

    for (GlobalMethodPool::iterator M = MethodPool.begin(),
                                    MEnd = MethodPool.end();
         M != MEnd; ++M) {
      for (ObjCMethodList *MethList = &M->second.second;
           MethList && MethList->Method; 
           MethList = MethList->Next) {
        if (!isAcceptableObjCMethod(MethList->Method, MK_Any, SelIdents, 
                                    NumSelIdents))
          continue;

        Result R(MethList->Method, 0);
        R.StartParameter = NumSelIdents;
        R.AllParametersAreInformative = false;
        Results.MaybeAddResult(R, CurContext);
      }
    }
  }

  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
}

void Sema::CodeCompleteObjCInstanceMessage(Scope *S, ExprTy *Receiver,
                                           IdentifierInfo **SelIdents,
                                           unsigned NumSelIdents) {
  typedef CodeCompletionResult Result;
  
  Expr *RecExpr = static_cast<Expr *>(Receiver);
  
  // If necessary, apply function/array conversion to the receiver.
  // C99 6.7.5.3p[7,8].
  DefaultFunctionArrayLvalueConversion(RecExpr);
  QualType ReceiverType = RecExpr->getType();
  
  // Build the set of methods we can see.
  ResultBuilder Results(*this);
  Results.EnterNewScope();

  // If we're messaging an expression with type "id" or "Class", check
  // whether we know something special about the receiver that allows
  // us to assume a more-specific receiver type.
  if (ReceiverType->isObjCIdType() || ReceiverType->isObjCClassType())
    if (ObjCInterfaceDecl *IFace = GetAssumedMessageSendExprType(RecExpr))
      ReceiverType = Context.getObjCObjectPointerType(
                                          Context.getObjCInterfaceType(IFace));
  
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
  // Handle messages to "id".
  else if (ReceiverType->isObjCIdType()) {
    // We're messaging "id", so provide all instance methods we know
    // about as code-completion results.

    // If we have an external source, load the entire class method
    // pool from the AST file.
    if (ExternalSource) {
      for (uint32_t I = 0, N = ExternalSource->GetNumExternalSelectors();
           I != N; ++I) {
        Selector Sel = ExternalSource->GetExternalSelector(I);
        if (Sel.isNull() || MethodPool.count(Sel))
          continue;

        ReadMethodPool(Sel);
      }
    }

    for (GlobalMethodPool::iterator M = MethodPool.begin(),
                                    MEnd = MethodPool.end();
         M != MEnd; ++M) {
      for (ObjCMethodList *MethList = &M->second.first;
           MethList && MethList->Method; 
           MethList = MethList->Next) {
        if (!isAcceptableObjCMethod(MethList->Method, MK_Any, SelIdents, 
                                    NumSelIdents))
          continue;

        Result R(MethList->Method, 0);
        R.StartParameter = NumSelIdents;
        R.AllParametersAreInformative = false;
        Results.MaybeAddResult(R, CurContext);
      }
    }
  }

  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
}

void Sema::CodeCompleteObjCForCollection(Scope *S, 
                                         DeclGroupPtrTy IterationVar) {
  CodeCompleteExpressionData Data;
  Data.ObjCCollection = true;
  
  if (IterationVar.getAsOpaquePtr()) {
    DeclGroupRef DG = IterationVar.getAsVal<DeclGroupRef>();
    for (DeclGroupRef::iterator I = DG.begin(), End = DG.end(); I != End; ++I) {
      if (*I)
        Data.IgnoreDecls.push_back(*I);
    }
  }
  
  CodeCompleteExpression(S, Data);
}

/// \brief Add all of the protocol declarations that we find in the given
/// (translation unit) context.
static void AddProtocolResults(DeclContext *Ctx, DeclContext *CurContext,
                               bool OnlyForwardDeclarations,
                               ResultBuilder &Results) {
  typedef CodeCompletionResult Result;
  
  for (DeclContext::decl_iterator D = Ctx->decls_begin(), 
                               DEnd = Ctx->decls_end();
       D != DEnd; ++D) {
    // Record any protocols we find.
    if (ObjCProtocolDecl *Proto = dyn_cast<ObjCProtocolDecl>(*D))
      if (!OnlyForwardDeclarations || Proto->isForwardDecl())
        Results.AddResult(Result(Proto, 0), CurContext, 0, false);

    // Record any forward-declared protocols we find.
    if (ObjCForwardProtocolDecl *Forward
          = dyn_cast<ObjCForwardProtocolDecl>(*D)) {
      for (ObjCForwardProtocolDecl::protocol_iterator 
             P = Forward->protocol_begin(),
             PEnd = Forward->protocol_end();
           P != PEnd; ++P)
        if (!OnlyForwardDeclarations || (*P)->isForwardDecl())
          Results.AddResult(Result(*P, 0), CurContext, 0, false);
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
    if (ObjCProtocolDecl *Protocol = LookupProtocol(Protocols[I].first,
                                                    Protocols[I].second))
      Results.Ignore(Protocol);

  // Add all protocols.
  AddProtocolResults(Context.getTranslationUnitDecl(), CurContext, false,
                     Results);

  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_ObjCProtocolName,
                            Results.data(),Results.size());
}

void Sema::CodeCompleteObjCProtocolDecl(Scope *) {
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  
  // Add all protocols.
  AddProtocolResults(Context.getTranslationUnitDecl(), CurContext, true,
                     Results);

  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_ObjCProtocolName,
                            Results.data(),Results.size());
}

/// \brief Add all of the Objective-C interface declarations that we find in
/// the given (translation unit) context.
static void AddInterfaceResults(DeclContext *Ctx, DeclContext *CurContext,
                                bool OnlyForwardDeclarations,
                                bool OnlyUnimplemented,
                                ResultBuilder &Results) {
  typedef CodeCompletionResult Result;
  
  for (DeclContext::decl_iterator D = Ctx->decls_begin(), 
                               DEnd = Ctx->decls_end();
       D != DEnd; ++D) {
    // Record any interfaces we find.
    if (ObjCInterfaceDecl *Class = dyn_cast<ObjCInterfaceDecl>(*D))
      if ((!OnlyForwardDeclarations || Class->isForwardDecl()) &&
          (!OnlyUnimplemented || !Class->getImplementation()))
        Results.AddResult(Result(Class, 0), CurContext, 0, false);

    // Record any forward-declared interfaces we find.
    if (ObjCClassDecl *Forward = dyn_cast<ObjCClassDecl>(*D)) {
      for (ObjCClassDecl::iterator C = Forward->begin(), CEnd = Forward->end();
           C != CEnd; ++C)
        if ((!OnlyForwardDeclarations || C->getInterface()->isForwardDecl()) &&
            (!OnlyUnimplemented || !C->getInterface()->getImplementation()))
          Results.AddResult(Result(C->getInterface(), 0), CurContext,
                            0, false);
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
  HandleCodeCompleteResults(this, CodeCompleter,
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
}

void Sema::CodeCompleteObjCSuperclass(Scope *S, IdentifierInfo *ClassName,
                                      SourceLocation ClassNameLoc) { 
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  
  // Make sure that we ignore the class we're currently defining.
  NamedDecl *CurClass
    = LookupSingleName(TUScope, ClassName, ClassNameLoc, LookupOrdinaryName);
  if (CurClass && isa<ObjCInterfaceDecl>(CurClass))
    Results.Ignore(CurClass);

  // Add all classes.
  AddInterfaceResults(Context.getTranslationUnitDecl(), CurContext, false,
                      false, Results);

  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
}

void Sema::CodeCompleteObjCImplementationDecl(Scope *S) { 
  ResultBuilder Results(*this);
  Results.EnterNewScope();

  // Add all unimplemented classes.
  AddInterfaceResults(Context.getTranslationUnitDecl(), CurContext, false,
                      true, Results);

  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
}

void Sema::CodeCompleteObjCInterfaceCategory(Scope *S, 
                                             IdentifierInfo *ClassName,
                                             SourceLocation ClassNameLoc) {
  typedef CodeCompletionResult Result;
  
  ResultBuilder Results(*this);
  
  // Ignore any categories we find that have already been implemented by this
  // interface.
  llvm::SmallPtrSet<IdentifierInfo *, 16> CategoryNames;
  NamedDecl *CurClass
    = LookupSingleName(TUScope, ClassName, ClassNameLoc, LookupOrdinaryName);
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
        Results.AddResult(Result(Category, 0), CurContext, 0, false);
  Results.ExitScope();
  
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());  
}

void Sema::CodeCompleteObjCImplementationCategory(Scope *S, 
                                                  IdentifierInfo *ClassName,
                                                  SourceLocation ClassNameLoc) {
  typedef CodeCompletionResult Result;
  
  // Find the corresponding interface. If we couldn't find the interface, the
  // program itself is ill-formed. However, we'll try to be helpful still by
  // providing the list of all of the categories we know about.
  NamedDecl *CurClass
    = LookupSingleName(TUScope, ClassName, ClassNameLoc, LookupOrdinaryName);
  ObjCInterfaceDecl *Class = dyn_cast_or_null<ObjCInterfaceDecl>(CurClass);
  if (!Class)
    return CodeCompleteObjCInterfaceCategory(S, ClassName, ClassNameLoc);
    
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
        Results.AddResult(Result(Category, 0), CurContext, 0, false);
    
    Class = Class->getSuperClass();
    IgnoreImplemented = false;
  }
  Results.ExitScope();
  
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());  
}

void Sema::CodeCompleteObjCPropertyDefinition(Scope *S, Decl *ObjCImpDecl) {
  typedef CodeCompletionResult Result;
  ResultBuilder Results(*this);

  // Figure out where this @synthesize lives.
  ObjCContainerDecl *Container
    = dyn_cast_or_null<ObjCContainerDecl>(ObjCImpDecl);
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
  
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());  
}

void Sema::CodeCompleteObjCPropertySynthesizeIvar(Scope *S, 
                                                  IdentifierInfo *PropertyName,
                                                  Decl *ObjCImpDecl) {
  typedef CodeCompletionResult Result;
  ResultBuilder Results(*this);

  // Figure out where this @synthesize lives.
  ObjCContainerDecl *Container
    = dyn_cast_or_null<ObjCContainerDecl>(ObjCImpDecl);
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
      Results.AddResult(Result(*IVar, 0), CurContext, 0, false);
  }
  Results.ExitScope();
  
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
}

// Mapping from selectors to the methods that implement that selector, along
// with the "in original class" flag.
typedef llvm::DenseMap<Selector, std::pair<ObjCMethodDecl *, bool> > 
  KnownMethodsMap;

/// \brief Find all of the methods that reside in the given container
/// (and its superclasses, protocols, etc.) that meet the given
/// criteria. Insert those methods into the map of known methods,
/// indexed by selector so they can be easily found.
static void FindImplementableMethods(ASTContext &Context,
                                     ObjCContainerDecl *Container,
                                     bool WantInstanceMethods,
                                     QualType ReturnType,
                                     bool IsInImplementation,
                                     KnownMethodsMap &KnownMethods,
                                     bool InOriginalClass = true) {
  if (ObjCInterfaceDecl *IFace = dyn_cast<ObjCInterfaceDecl>(Container)) {
    // Recurse into protocols.
    const ObjCList<ObjCProtocolDecl> &Protocols
      = IFace->getReferencedProtocols();
    for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
           E = Protocols.end(); 
         I != E; ++I)
      FindImplementableMethods(Context, *I, WantInstanceMethods, ReturnType,
                               IsInImplementation, KnownMethods,
                               InOriginalClass);

    // If we're not in the implementation of a class, also visit the
    // superclass.
    if (!IsInImplementation && IFace->getSuperClass())
      FindImplementableMethods(Context, IFace->getSuperClass(), 
                               WantInstanceMethods, ReturnType,
                               IsInImplementation, KnownMethods,
                               false);

    // Add methods from any class extensions (but not from categories;
    // those should go into category implementations).
    for (const ObjCCategoryDecl *Cat = IFace->getFirstClassExtension(); Cat;
         Cat = Cat->getNextClassExtension())
      FindImplementableMethods(Context, const_cast<ObjCCategoryDecl*>(Cat), 
                               WantInstanceMethods, ReturnType,
                               IsInImplementation, KnownMethods,
                               InOriginalClass);      
  }

  if (ObjCCategoryDecl *Category = dyn_cast<ObjCCategoryDecl>(Container)) {
    // Recurse into protocols.
    const ObjCList<ObjCProtocolDecl> &Protocols
      = Category->getReferencedProtocols();
    for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
           E = Protocols.end(); 
         I != E; ++I)
      FindImplementableMethods(Context, *I, WantInstanceMethods, ReturnType,
                               IsInImplementation, KnownMethods,
                               InOriginalClass);
  }

  if (ObjCProtocolDecl *Protocol = dyn_cast<ObjCProtocolDecl>(Container)) {
    // Recurse into protocols.
    const ObjCList<ObjCProtocolDecl> &Protocols
      = Protocol->getReferencedProtocols();
    for (ObjCList<ObjCProtocolDecl>::iterator I = Protocols.begin(),
           E = Protocols.end(); 
         I != E; ++I)
      FindImplementableMethods(Context, *I, WantInstanceMethods, ReturnType,
                               IsInImplementation, KnownMethods, false);
  }

  // Add methods in this container. This operation occurs last because
  // we want the methods from this container to override any methods
  // we've previously seen with the same selector.
  for (ObjCContainerDecl::method_iterator M = Container->meth_begin(),
                                       MEnd = Container->meth_end();
       M != MEnd; ++M) {
    if ((*M)->isInstanceMethod() == WantInstanceMethods) {
      if (!ReturnType.isNull() &&
          !Context.hasSameUnqualifiedType(ReturnType, (*M)->getResultType()))
        continue;

      KnownMethods[(*M)->getSelector()] = std::make_pair(*M, InOriginalClass);
    }
  }
}

void Sema::CodeCompleteObjCMethodDecl(Scope *S, 
                                      bool IsInstanceMethod,
                                      ParsedType ReturnTy,
                                      Decl *IDecl) {
  // Determine the return type of the method we're declaring, if
  // provided.
  QualType ReturnType = GetTypeFromParser(ReturnTy);

  // Determine where we should start searching for methods, and where we 
  ObjCContainerDecl *SearchDecl = 0, *CurrentDecl = 0;
  bool IsInImplementation = false;
  if (Decl *D = IDecl) {
    if (ObjCImplementationDecl *Impl = dyn_cast<ObjCImplementationDecl>(D)) {
      SearchDecl = Impl->getClassInterface();
      CurrentDecl = Impl;
      IsInImplementation = true;
    } else if (ObjCCategoryImplDecl *CatImpl 
                                       = dyn_cast<ObjCCategoryImplDecl>(D)) {
      SearchDecl = CatImpl->getCategoryDecl();
      CurrentDecl = CatImpl;
      IsInImplementation = true;
    } else {
      SearchDecl = dyn_cast<ObjCContainerDecl>(D);
      CurrentDecl = SearchDecl;
    }
  }

  if (!SearchDecl && S) {
    if (DeclContext *DC = static_cast<DeclContext *>(S->getEntity())) {
      SearchDecl = dyn_cast<ObjCContainerDecl>(DC);
      CurrentDecl = SearchDecl;
    }
  }

  if (!SearchDecl || !CurrentDecl) {
    HandleCodeCompleteResults(this, CodeCompleter, 
                              CodeCompletionContext::CCC_Other,
                              0, 0);
    return;
  }
    
  // Find all of the methods that we could declare/implement here.
  KnownMethodsMap KnownMethods;
  FindImplementableMethods(Context, SearchDecl, IsInstanceMethod, 
                           ReturnType, IsInImplementation, KnownMethods);
  
  // Erase any methods that have already been declared or
  // implemented here.
  for (ObjCContainerDecl::method_iterator M = CurrentDecl->meth_begin(),
                                       MEnd = CurrentDecl->meth_end();
       M != MEnd; ++M) {
    if ((*M)->isInstanceMethod() != IsInstanceMethod)
      continue;
    
    KnownMethodsMap::iterator Pos = KnownMethods.find((*M)->getSelector());
    if (Pos != KnownMethods.end())
      KnownMethods.erase(Pos);
  }

  // Add declarations or definitions for each of the known methods.
  typedef CodeCompletionResult Result;
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  PrintingPolicy Policy(Context.PrintingPolicy);
  Policy.AnonymousTagLocations = false;
  for (KnownMethodsMap::iterator M = KnownMethods.begin(), 
                              MEnd = KnownMethods.end();
       M != MEnd; ++M) {
    ObjCMethodDecl *Method = M->second.first;
    CodeCompletionString *Pattern = new CodeCompletionString;
    
    // If the result type was not already provided, add it to the
    // pattern as (type).
    if (ReturnType.isNull()) {
      std::string TypeStr;
      Method->getResultType().getAsStringInternal(TypeStr, Policy);
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddTextChunk(TypeStr);
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
    }

    Selector Sel = Method->getSelector();

    // Add the first part of the selector to the pattern.
    Pattern->AddTypedTextChunk(Sel.getIdentifierInfoForSlot(0)->getName());

    // Add parameters to the pattern.
    unsigned I = 0;
    for (ObjCMethodDecl::param_iterator P = Method->param_begin(), 
                                     PEnd = Method->param_end();
         P != PEnd; (void)++P, ++I) {
      // Add the part of the selector name.
      if (I == 0)
        Pattern->AddChunk(CodeCompletionString::CK_Colon);
      else if (I < Sel.getNumArgs()) {
        Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
        Pattern->AddTextChunk(Sel.getIdentifierInfoForSlot(I)->getName());
        Pattern->AddChunk(CodeCompletionString::CK_Colon);
      } else
        break;

      // Add the parameter type.
      std::string TypeStr;
      (*P)->getOriginalType().getAsStringInternal(TypeStr, Policy);
      Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
      Pattern->AddTextChunk(TypeStr);
      Pattern->AddChunk(CodeCompletionString::CK_RightParen);
      
      if (IdentifierInfo *Id = (*P)->getIdentifier())
        Pattern->AddTextChunk(Id->getName());
    }

    if (Method->isVariadic()) {
      if (Method->param_size() > 0)
        Pattern->AddChunk(CodeCompletionString::CK_Comma);
      Pattern->AddTextChunk("...");
    }

    if (IsInImplementation && Results.includeCodePatterns()) {
      // We will be defining the method here, so add a compound statement.
      Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
      Pattern->AddChunk(CodeCompletionString::CK_LeftBrace);
      Pattern->AddChunk(CodeCompletionString::CK_VerticalSpace);
      if (!Method->getResultType()->isVoidType()) {
        // If the result type is not void, add a return clause.
        Pattern->AddTextChunk("return");
        Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
        Pattern->AddPlaceholderChunk("expression");
        Pattern->AddChunk(CodeCompletionString::CK_SemiColon);
      } else
        Pattern->AddPlaceholderChunk("statements");
        
      Pattern->AddChunk(CodeCompletionString::CK_VerticalSpace);
      Pattern->AddChunk(CodeCompletionString::CK_RightBrace);
    }

    unsigned Priority = CCP_CodePattern;
    if (!M->second.second)
      Priority += CCD_InBaseClass;
    
    Results.AddResult(Result(Pattern, Priority, 
                             Method->isInstanceMethod()
                               ? CXCursor_ObjCInstanceMethodDecl
                               : CXCursor_ObjCClassMethodDecl));
  }

  Results.ExitScope();
  
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
}

void Sema::CodeCompleteObjCMethodDeclSelector(Scope *S, 
                                              bool IsInstanceMethod,
                                              bool AtParameterName,
                                              ParsedType ReturnTy,
                                              IdentifierInfo **SelIdents,
                                              unsigned NumSelIdents) {
  // If we have an external source, load the entire class method
  // pool from the AST file.
  if (ExternalSource) {
    for (uint32_t I = 0, N = ExternalSource->GetNumExternalSelectors();
         I != N; ++I) {
      Selector Sel = ExternalSource->GetExternalSelector(I);
      if (Sel.isNull() || MethodPool.count(Sel))
        continue;

      ReadMethodPool(Sel);
    }
  }

  // Build the set of methods we can see.
  typedef CodeCompletionResult Result;
  ResultBuilder Results(*this);
  
  if (ReturnTy)
    Results.setPreferredType(GetTypeFromParser(ReturnTy).getNonReferenceType());

  Results.EnterNewScope();  
  for (GlobalMethodPool::iterator M = MethodPool.begin(),
                                  MEnd = MethodPool.end();
       M != MEnd; ++M) {
    for (ObjCMethodList *MethList = IsInstanceMethod ? &M->second.first :
                                                       &M->second.second;
         MethList && MethList->Method; 
         MethList = MethList->Next) {
      if (!isAcceptableObjCMethod(MethList->Method, MK_Any, SelIdents, 
                                  NumSelIdents))
        continue;
      
      if (AtParameterName) {
        // Suggest parameter names we've seen before.
        if (NumSelIdents && NumSelIdents <= MethList->Method->param_size()) {
          ParmVarDecl *Param = MethList->Method->param_begin()[NumSelIdents-1];
          if (Param->getIdentifier()) {
            CodeCompletionString *Pattern = new CodeCompletionString;
            Pattern->AddTypedTextChunk(Param->getIdentifier()->getName());
            Results.AddResult(Pattern);
          }
        }
        
        continue;
      }
      
      Result R(MethList->Method, 0);
      R.StartParameter = NumSelIdents;
      R.AllParametersAreInformative = false;
      R.DeclaringEntity = true;
      Results.MaybeAddResult(R, CurContext);
    }
  }
  
  Results.ExitScope();
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_Other,
                            Results.data(),Results.size());
}

void Sema::CodeCompletePreprocessorDirective(bool InConditional) {
  ResultBuilder Results(*this);
  Results.EnterNewScope();
  
  // #if <condition>
  CodeCompletionString *Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("if");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("condition");
  Results.AddResult(Pattern);
  
  // #ifdef <macro>
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("ifdef");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("macro");
  Results.AddResult(Pattern);

  // #ifndef <macro>
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("ifndef");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("macro");
  Results.AddResult(Pattern);

  if (InConditional) {
    // #elif <condition>
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("elif");
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddPlaceholderChunk("condition");
    Results.AddResult(Pattern);

    // #else
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("else");
    Results.AddResult(Pattern);

    // #endif
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("endif");
    Results.AddResult(Pattern);
  }
  
  // #include "header"
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("include");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddTextChunk("\"");
  Pattern->AddPlaceholderChunk("header");
  Pattern->AddTextChunk("\"");
  Results.AddResult(Pattern);

  // #include <header>
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("include");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddTextChunk("<");
  Pattern->AddPlaceholderChunk("header");
  Pattern->AddTextChunk(">");
  Results.AddResult(Pattern);
  
  // #define <macro>
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("define");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("macro");
  Results.AddResult(Pattern);
  
  // #define <macro>(<args>)
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("define");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("macro");
  Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
  Pattern->AddPlaceholderChunk("args");
  Pattern->AddChunk(CodeCompletionString::CK_RightParen);
  Results.AddResult(Pattern);
  
  // #undef <macro>
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("undef");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("macro");
  Results.AddResult(Pattern);

  // #line <number>
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("line");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("number");
  Results.AddResult(Pattern);
  
  // #line <number> "filename"
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("line");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("number");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddTextChunk("\"");
  Pattern->AddPlaceholderChunk("filename");
  Pattern->AddTextChunk("\"");
  Results.AddResult(Pattern);
  
  // #error <message>
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("error");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("message");
  Results.AddResult(Pattern);

  // #pragma <arguments>
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("pragma");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("arguments");
  Results.AddResult(Pattern);

  if (getLangOptions().ObjC1) {
    // #import "header"
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("import");
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddTextChunk("\"");
    Pattern->AddPlaceholderChunk("header");
    Pattern->AddTextChunk("\"");
    Results.AddResult(Pattern);
    
    // #import <header>
    Pattern = new CodeCompletionString;
    Pattern->AddTypedTextChunk("import");
    Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Pattern->AddTextChunk("<");
    Pattern->AddPlaceholderChunk("header");
    Pattern->AddTextChunk(">");
    Results.AddResult(Pattern);
  }
  
  // #include_next "header"
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("include_next");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddTextChunk("\"");
  Pattern->AddPlaceholderChunk("header");
  Pattern->AddTextChunk("\"");
  Results.AddResult(Pattern);
  
  // #include_next <header>
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("include_next");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddTextChunk("<");
  Pattern->AddPlaceholderChunk("header");
  Pattern->AddTextChunk(">");
  Results.AddResult(Pattern);

  // #warning <message>
  Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("warning");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddPlaceholderChunk("message");
  Results.AddResult(Pattern);

  // Note: #ident and #sccs are such crazy anachronisms that we don't provide
  // completions for them. And __include_macros is a Clang-internal extension
  // that we don't want to encourage anyone to use.

  // FIXME: we don't support #assert or #unassert, so don't suggest them.
  Results.ExitScope();
  
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_PreprocessorDirective,
                            Results.data(), Results.size());
}

void Sema::CodeCompleteInPreprocessorConditionalExclusion(Scope *S) {
  CodeCompleteOrdinaryName(S,
                           S->getFnParent()? Action::PCC_RecoveryInFunction 
                                           : Action::PCC_Namespace);
}

void Sema::CodeCompletePreprocessorMacroName(bool IsDefinition) {
  ResultBuilder Results(*this);
  if (!IsDefinition && (!CodeCompleter || CodeCompleter->includeMacros())) {
    // Add just the names of macros, not their arguments.    
    Results.EnterNewScope();
    for (Preprocessor::macro_iterator M = PP.macro_begin(), 
                                   MEnd = PP.macro_end();
         M != MEnd; ++M) {
      CodeCompletionString *Pattern = new CodeCompletionString;
      Pattern->AddTypedTextChunk(M->first->getName());
      Results.AddResult(Pattern);
    }
    Results.ExitScope();
  } else if (IsDefinition) {
    // FIXME: Can we detect when the user just wrote an include guard above?
  }
  
  HandleCodeCompleteResults(this, CodeCompleter, 
                      IsDefinition? CodeCompletionContext::CCC_MacroName
                                  : CodeCompletionContext::CCC_MacroNameUse,
                            Results.data(), Results.size()); 
}

void Sema::CodeCompletePreprocessorExpression() {
  ResultBuilder Results(*this);
  
  if (!CodeCompleter || CodeCompleter->includeMacros())
    AddMacroResults(PP, Results);
  
    // defined (<macro>)
  Results.EnterNewScope();
  CodeCompletionString *Pattern = new CodeCompletionString;
  Pattern->AddTypedTextChunk("defined");
  Pattern->AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Pattern->AddChunk(CodeCompletionString::CK_LeftParen);
  Pattern->AddPlaceholderChunk("macro");
  Pattern->AddChunk(CodeCompletionString::CK_RightParen);
  Results.AddResult(Pattern);
  Results.ExitScope();
  
  HandleCodeCompleteResults(this, CodeCompleter, 
                            CodeCompletionContext::CCC_PreprocessorExpression,
                            Results.data(), Results.size()); 
}

void Sema::CodeCompletePreprocessorMacroArgument(Scope *S,
                                                 IdentifierInfo *Macro,
                                                 MacroInfo *MacroInfo,
                                                 unsigned Argument) {
  // FIXME: In the future, we could provide "overload" results, much like we
  // do for function calls.
  
  CodeCompleteOrdinaryName(S,
                           S->getFnParent()? Action::PCC_RecoveryInFunction 
                                           : Action::PCC_Namespace);
}

void Sema::CodeCompleteNaturalLanguage() {
  HandleCodeCompleteResults(this, CodeCompleter,
                            CodeCompletionContext::CCC_NaturalLanguage,
                            0, 0);
}

void Sema::GatherGlobalCodeCompletions(
                 llvm::SmallVectorImpl<CodeCompletionResult> &Results) {
  ResultBuilder Builder(*this);

  if (!CodeCompleter || CodeCompleter->includeGlobals()) {
    CodeCompletionDeclConsumer Consumer(Builder, 
                                        Context.getTranslationUnitDecl());
    LookupVisibleDecls(Context.getTranslationUnitDecl(), LookupAnyName, 
                       Consumer);
  }
  
  if (!CodeCompleter || CodeCompleter->includeMacros())
    AddMacroResults(PP, Builder);
  
  Results.clear();
  Results.insert(Results.end(), 
                 Builder.data(), Builder.data() + Builder.size());
}
