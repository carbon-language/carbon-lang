//===---- CodeCompleteConsumer.h - Code Completion Interface ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the CodeCompleteConsumer class.
//
//===----------------------------------------------------------------------===//
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Parse/Scope.h"
#include "clang/Lex/Preprocessor.h"
#include "Sema.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstring>
#include <functional>
using namespace clang;

//===----------------------------------------------------------------------===//
// Code completion string implementation
//===----------------------------------------------------------------------===//
CodeCompletionString::Chunk
CodeCompletionString::Chunk::CreateText(const char *Text) {
  Chunk Result;
  Result.Kind = CK_Text;
  char *New = new char [std::strlen(Text) + 1];
  std::strcpy(New, Text);
  Result.Text = New;
  return Result;  
}

CodeCompletionString::Chunk 
CodeCompletionString::Chunk::CreateOptional(
                                 std::auto_ptr<CodeCompletionString> Optional) {
  Chunk Result;
  Result.Kind = CK_Optional;
  Result.Optional = Optional.release();
  return Result;
}

CodeCompletionString::Chunk 
CodeCompletionString::Chunk::CreatePlaceholder(const char *Placeholder) {
  Chunk Result;
  Result.Kind = CK_Placeholder;
  char *New = new char [std::strlen(Placeholder) + 1];
  std::strcpy(New, Placeholder);
  Result.Placeholder = New;
  return Result;
}

void
CodeCompletionString::Chunk::Destroy() {
  switch (Kind) {
  case CK_Text: delete [] Text; break;
  case CK_Optional: delete Optional; break;
  case CK_Placeholder: delete [] Placeholder; break;
  }
}

CodeCompletionString::~CodeCompletionString() {
  std::for_each(Chunks.begin(), Chunks.end(), 
                std::mem_fun_ref(&Chunk::Destroy));
}

std::string CodeCompletionString::getAsString() const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
                          
  for (iterator C = begin(), CEnd = end(); C != CEnd; ++C) {
    switch (C->Kind) {
    case CK_Text: OS << C->Text; break;
    case CK_Optional: OS << "{#" << C->Optional->getAsString() << "#}"; break;
    case CK_Placeholder: OS << "<#" << C->Placeholder << "#>"; break;
    }
  }
  
  return Result;
}

//===----------------------------------------------------------------------===//
// Code completion consumer implementation
//===----------------------------------------------------------------------===//

CodeCompleteConsumer::CodeCompleteConsumer(Sema &S) : SemaRef(S) {
  SemaRef.setCodeCompleteConsumer(this);
}

CodeCompleteConsumer::~CodeCompleteConsumer() {
  SemaRef.setCodeCompleteConsumer(0);
}

void 
CodeCompleteConsumer::CodeCompleteMemberReferenceExpr(Scope *S, 
                                                      QualType BaseType,
                                                      bool IsArrow) {
  if (IsArrow) {
    if (const PointerType *Ptr = BaseType->getAs<PointerType>())
      BaseType = Ptr->getPointeeType();
    else if (BaseType->isObjCObjectPointerType())
    /*Do nothing*/ ;
    else
      return;
  }
  
  ResultSet Results(*this);
  unsigned NextRank = 0;
  
  if (const RecordType *Record = BaseType->getAs<RecordType>()) {
    NextRank = CollectMemberLookupResults(Record->getDecl(), NextRank, Results);

    if (getSema().getLangOptions().CPlusPlus) {
      if (!Results.empty())
        // The "template" keyword can follow "->" or "." in the grammar.
        Results.MaybeAddResult(Result("template", NextRank++));

      // We could have the start of a nested-name-specifier. Add those
      // results as well.
      Results.setFilter(&CodeCompleteConsumer::IsNestedNameSpecifier);
      CollectLookupResults(S, NextRank, Results);
    }

    // Hand off the results found for code completion.
    ProcessCodeCompleteResults(Results.data(), Results.size());
    
    // We're done!
    return;
  }
}

void CodeCompleteConsumer::CodeCompleteTag(Scope *S, ElaboratedType::TagKind TK) {
  ResultSet::LookupFilter Filter = 0;
  switch (TK) {
  case ElaboratedType::TK_enum:
    Filter = &CodeCompleteConsumer::IsEnum;
    break;
    
  case ElaboratedType::TK_class:
  case ElaboratedType::TK_struct:
    Filter = &CodeCompleteConsumer::IsClassOrStruct;
    break;
    
  case ElaboratedType::TK_union:
    Filter = &CodeCompleteConsumer::IsUnion;
    break;
  }
  
  ResultSet Results(*this, Filter);
  unsigned NextRank = CollectLookupResults(S, 0, Results);
  
  if (getSema().getLangOptions().CPlusPlus) {
    // We could have the start of a nested-name-specifier. Add those
    // results as well.
    Results.setFilter(&CodeCompleteConsumer::IsNestedNameSpecifier);
    CollectLookupResults(S, NextRank, Results);
  }
  
  ProcessCodeCompleteResults(Results.data(), Results.size());
}

void 
CodeCompleteConsumer::CodeCompleteQualifiedId(Scope *S, 
                                              NestedNameSpecifier *NNS,
                                              bool EnteringContext) {
  CXXScopeSpec SS;
  SS.setScopeRep(NNS);
  DeclContext *Ctx = getSema().computeDeclContext(SS, EnteringContext);
  if (!Ctx)
    return;
  
  ResultSet Results(*this);
  unsigned NextRank = CollectMemberLookupResults(Ctx, 0, Results);
  
  // The "template" keyword can follow "::" in the grammar
  if (!Results.empty())
    Results.MaybeAddResult(Result("template", NextRank));
  
  ProcessCodeCompleteResults(Results.data(), Results.size());
}

void CodeCompleteConsumer::CodeCompleteUsing(Scope *S) { 
  ResultSet Results(*this, &CodeCompleteConsumer::IsNestedNameSpecifier);
  
  // If we aren't in class scope, we could see the "namespace" keyword.
  if (!S->isClassScope())
    Results.MaybeAddResult(Result("namespace", 0));
    
  // After "using", we can see anything that would start a 
  // nested-name-specifier.
  CollectLookupResults(S, 0, Results);
  
  ProcessCodeCompleteResults(Results.data(), Results.size());
}

void CodeCompleteConsumer::CodeCompleteUsingDirective(Scope *S) { 
  // After "using namespace", we expect to see a namespace name or namespace
  // alias.
  ResultSet Results(*this, &CodeCompleteConsumer::IsNamespaceOrAlias);
  CollectLookupResults(S, 0, Results);
  ProcessCodeCompleteResults(Results.data(), Results.size());  
}

void CodeCompleteConsumer::CodeCompleteNamespaceDecl(Scope *S) { 
  ResultSet Results(*this, &CodeCompleteConsumer::IsNamespace);
  DeclContext *Ctx = (DeclContext *)S->getEntity();
  if (!S->getParent())
    Ctx = getSema().Context.getTranslationUnitDecl();

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
    for (std::map<NamespaceDecl *, NamespaceDecl *>::iterator 
          NS = OrigToLatest.begin(), NSEnd = OrigToLatest.end();
         NS != NSEnd; ++NS)
      Results.MaybeAddResult(Result(NS->second, 0));
  }
  
  ProcessCodeCompleteResults(Results.data(), Results.size());  
}

void CodeCompleteConsumer::CodeCompleteNamespaceAliasDecl(Scope *S) { 
  // After "namespace", we expect to see a namespace  or alias.
  ResultSet Results(*this, &CodeCompleteConsumer::IsNamespaceOrAlias);
  CollectLookupResults(S, 0, Results);
  ProcessCodeCompleteResults(Results.data(), Results.size());  
}

void CodeCompleteConsumer::CodeCompleteOperatorName(Scope *S) {
  ResultSet Results(*this, &CodeCompleteConsumer::IsType);
  
  // Add the names of overloadable operators.
#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly)      \
  if (std::strcmp(Spelling, "?"))                                                  \
    Results.MaybeAddResult(Result(Spelling, 0));
#include "clang/Basic/OperatorKinds.def"
  
  // Add any type names visible from the current scope
  unsigned NextRank = CollectLookupResults(S, 0, Results);
  
  // Add any type specifiers
  AddTypeSpecifierResults(0, Results);
  
  // Add any nested-name-specifiers
  Results.setFilter(&CodeCompleteConsumer::IsNestedNameSpecifier);
  CollectLookupResults(S, NextRank + 1, Results);

  ProcessCodeCompleteResults(Results.data(), Results.size());  
}

void CodeCompleteConsumer::ResultSet::MaybeAddResult(Result R) {
  if (R.Kind != Result::RK_Declaration) {
    // For non-declaration results, just add the result.
    Results.push_back(R);
    return;
  }

  // Look through using declarations.
  if (UsingDecl *Using = dyn_cast<UsingDecl>(R.Declaration))
    return MaybeAddResult(Result(Using->getTargetDecl(), R.Rank));
  
  // Handle each declaration in an overload set separately.
  if (OverloadedFunctionDecl *Ovl 
        = dyn_cast<OverloadedFunctionDecl>(R.Declaration)) {
    for (OverloadedFunctionDecl::function_iterator F = Ovl->function_begin(),
                                                FEnd = Ovl->function_end();
         F != FEnd; ++F)
      MaybeAddResult(Result(*F, R.Rank));
    
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
  if (Filter && !(Completer.*Filter)(R.Declaration))
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
      if (Completer.canHiddenResultBeFound(R.Declaration, I->second.first)) {
        // Note that this result was hidden.
        R.Hidden = true;
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
  
  // Insert this result into the set of results and into the current shadow
  // map.
  SMap.insert(std::make_pair(R.Declaration->getDeclName(),
                             std::make_pair(R.Declaration, Results.size())));
  Results.push_back(R);
}

/// \brief Enter into a new scope.
void CodeCompleteConsumer::ResultSet::EnterNewScope() {
  ShadowMaps.push_back(ShadowMap());
}

/// \brief Exit from the current scope.
void CodeCompleteConsumer::ResultSet::ExitScope() {
  ShadowMaps.pop_back();
}

// Find the next outer declaration context corresponding to this scope.
static DeclContext *findOuterContext(Scope *S) {
  for (S = S->getParent(); S; S = S->getParent())
    if (S->getEntity())
      return static_cast<DeclContext *>(S->getEntity())->getPrimaryContext();
  
  return 0;
}

/// \brief Collect the results of searching for declarations within the given
/// scope and its parent scopes.
///
/// \param S the scope in which we will start looking for declarations.
///
/// \param InitialRank the initial rank given to results in this scope.
/// Larger rank values will be used for results found in parent scopes.
unsigned CodeCompleteConsumer::CollectLookupResults(Scope *S, 
                                                    unsigned InitialRank,
                                                    ResultSet &Results) {
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
      
      NextRank = CollectMemberLookupResults(Ctx, NextRank + 1, Results);
    }
  } else if (!S->getParent()) {
    // Look into the translation unit scope. We walk through the translation
    // unit's declaration context, because the Scope itself won't have all of
    // the declarations if 
    NextRank = CollectMemberLookupResults(
                                    getSema().Context.getTranslationUnitDecl(), 
                                          NextRank + 1, Results);
  } else {
    // Walk through the declarations in this Scope.
    for (Scope::decl_iterator D = S->decl_begin(), DEnd = S->decl_end();
         D != DEnd; ++D) {
      if (NamedDecl *ND = dyn_cast<NamedDecl>((Decl *)((*D).get())))
        Results.MaybeAddResult(Result(ND, NextRank));        
    }
    
    NextRank = NextRank + 1;
  }
  
  // Lookup names in the parent scope.
  NextRank = CollectLookupResults(S->getParent(), NextRank, Results);
  Results.ExitScope();
  
  return NextRank;
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
unsigned CodeCompleteConsumer::CollectMemberLookupResults(DeclContext *Ctx, 
                                                          unsigned InitialRank, 
                                                          ResultSet &Results) {
  llvm::SmallPtrSet<DeclContext *, 16> Visited;
  return CollectMemberLookupResults(Ctx, InitialRank, Visited, Results);
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
/// \param Visited the set of declaration contexts that have already been
/// visited. Declaration contexts will only be visited once.
///
/// \param Results the result set that will be extended with any results
/// found within this declaration context (and, for a C++ class, its bases).
///
/// \returns the next higher rank value, after considering all of the
/// names within this declaration context.
unsigned CodeCompleteConsumer::CollectMemberLookupResults(DeclContext *Ctx, 
                                                          unsigned InitialRank,
                                 llvm::SmallPtrSet<DeclContext *, 16> &Visited,
                                                          ResultSet &Results) {
  // Make sure we don't visit the same context twice.
  if (!Visited.insert(Ctx->getPrimaryContext()))
    return InitialRank;
  
  // Enumerate all of the results in this context.
  Results.EnterNewScope();
  for (DeclContext *CurCtx = Ctx->getPrimaryContext(); CurCtx; 
       CurCtx = CurCtx->getNextContext()) {
    for (DeclContext::decl_iterator D = CurCtx->decls_begin(), 
                                 DEnd = CurCtx->decls_end();
         D != DEnd; ++D) {
      if (NamedDecl *ND = dyn_cast<NamedDecl>(*D))
        Results.MaybeAddResult(Result(ND, InitialRank));
    }
  }
  
  // Traverse the contexts of inherited classes.
  unsigned NextRank = InitialRank;
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
      NextRank = std::max(NextRank, 
                          CollectMemberLookupResults(Record->getDecl(), 
                                                     InitialRank + 1,
                                                     Visited,
                                                     Results));
    }
  }
  
  // FIXME: Look into base classes in Objective-C!

  Results.ExitScope();
  return NextRank;
}

/// \brief Determines whether the given declaration is suitable as the 
/// start of a C++ nested-name-specifier, e.g., a class or namespace.
bool CodeCompleteConsumer::IsNestedNameSpecifier(NamedDecl *ND) const {
  // Allow us to find class templates, too.
  if (ClassTemplateDecl *ClassTemplate = dyn_cast<ClassTemplateDecl>(ND))
    ND = ClassTemplate->getTemplatedDecl();
  
  return getSema().isAcceptableNestedNameSpecifier(ND);
}

/// \brief Determines whether the given declaration is an enumeration.
bool CodeCompleteConsumer::IsEnum(NamedDecl *ND) const {
  return isa<EnumDecl>(ND);
}

/// \brief Determines whether the given declaration is a class or struct.
bool CodeCompleteConsumer::IsClassOrStruct(NamedDecl *ND) const {
  // Allow us to find class templates, too.
  if (ClassTemplateDecl *ClassTemplate = dyn_cast<ClassTemplateDecl>(ND))
    ND = ClassTemplate->getTemplatedDecl();
  
  if (RecordDecl *RD = dyn_cast<RecordDecl>(ND))
    return RD->getTagKind() == TagDecl::TK_class ||
           RD->getTagKind() == TagDecl::TK_struct;
  
  return false;
}

/// \brief Determines whether the given declaration is a union.
bool CodeCompleteConsumer::IsUnion(NamedDecl *ND) const {
  // Allow us to find class templates, too.
  if (ClassTemplateDecl *ClassTemplate = dyn_cast<ClassTemplateDecl>(ND))
    ND = ClassTemplate->getTemplatedDecl();

  if (RecordDecl *RD = dyn_cast<RecordDecl>(ND))
    return RD->getTagKind() == TagDecl::TK_union;
  
  return false;
}

/// \brief Determines whether the given declaration is a namespace.
bool CodeCompleteConsumer::IsNamespace(NamedDecl *ND) const {
  return isa<NamespaceDecl>(ND);
}

/// \brief Determines whether the given declaration is a namespace or 
/// namespace alias.
bool CodeCompleteConsumer::IsNamespaceOrAlias(NamedDecl *ND) const {
  return isa<NamespaceDecl>(ND) || isa<NamespaceAliasDecl>(ND);
}

/// \brief Brief determines whether the given declaration is a namespace or
/// namespace alias.
bool CodeCompleteConsumer::IsType(NamedDecl *ND) const {
  return isa<TypeDecl>(ND);
}

namespace {
  struct VISIBILITY_HIDDEN SortCodeCompleteResult {
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
      
      // If only our C++ compiler did control-flow warnings properly.
      return false;
    }
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
bool CodeCompleteConsumer::canHiddenResultBeFound(NamedDecl *Hidden, 
                                                  NamedDecl *Visible) {
  // In C, there is no way to refer to a hidden name.
  if (!getSema().getLangOptions().CPlusPlus)
    return false;
  
  DeclContext *HiddenCtx = Hidden->getDeclContext()->getLookupContext();
  
  // There is no way to qualify a name declared in a function or method.
  if (HiddenCtx->isFunctionOrMethod())
    return false;

  // If the hidden and visible declarations are in different name-lookup
  // contexts, then we can qualify the name of the hidden declaration.
  // FIXME: Optionally compute the string needed to refer to the hidden
  // name.
  return HiddenCtx != Visible->getDeclContext()->getLookupContext();
}

/// \brief Add type specifiers for the current language as keyword results.
void CodeCompleteConsumer::AddTypeSpecifierResults(unsigned Rank, 
                                                   ResultSet &Results) {
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

  if (getSema().getLangOptions().C99) {
    // C99-specific
    Results.MaybeAddResult(Result("_Complex", Rank));
    Results.MaybeAddResult(Result("_Imaginary", Rank));
    Results.MaybeAddResult(Result("_Bool", Rank));
  }
  
  if (getSema().getLangOptions().CPlusPlus) {
    // C++-specific
    Results.MaybeAddResult(Result("bool", Rank));
    Results.MaybeAddResult(Result("class", Rank));
    Results.MaybeAddResult(Result("typename", Rank));
    Results.MaybeAddResult(Result("wchar_t", Rank));
    
    if (getSema().getLangOptions().CPlusPlus0x) {
      Results.MaybeAddResult(Result("char16_t", Rank));
      Results.MaybeAddResult(Result("char32_t", Rank));
      Results.MaybeAddResult(Result("decltype", Rank));
    }
  }
  
  // GNU extensions
  if (getSema().getLangOptions().GNUMode) {
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
}

/// \brief Add template parameter chunks to the given code completion string.
static void AddTemplateParameterChunks(ASTContext &Context,
                                       TemplateDecl *Template,
                                       CodeCompletionString *Result) {
  CodeCompletionString *CCStr = Result;
  bool FirstParameter = true;
  
  TemplateParameterList *Params = Template->getTemplateParameters();
  for (TemplateParameterList::iterator P = Params->begin(), 
                                    PEnd = Params->end();
       P != PEnd; ++P) {
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

/// \brief If possible, create a new code completion string for the given
/// result.
///
/// \returns Either a new, heap-allocated code completion string describing
/// how to use this result, or NULL to indicate that the string or name of the
/// result is all that is needed.
CodeCompletionString *
CodeCompleteConsumer::CreateCodeCompletionString(Result R) {
  if (R.Kind != Result::RK_Declaration)
    return 0;
  
  NamedDecl *ND = R.Declaration;

  if (FunctionDecl *Function = dyn_cast<FunctionDecl>(ND)) {
    CodeCompletionString *Result = new CodeCompletionString;
    Result->AddTextChunk(Function->getNameAsString().c_str());
    Result->AddTextChunk("(");
    AddFunctionParameterChunks(getSema().Context, Function, Result);
    Result->AddTextChunk(")");
    return Result;
  }
  
  if (FunctionTemplateDecl *FunTmpl = dyn_cast<FunctionTemplateDecl>(ND)) {
    // FIXME: We treat these like functions for now, but it would be far
    // better if we computed the template parameters that are non-deduced from
    // a call, then printed only those template parameters in "<...>" before
    // printing the function call arguments.
    CodeCompletionString *Result = new CodeCompletionString;
    FunctionDecl *Function = FunTmpl->getTemplatedDecl();
    Result->AddTextChunk(Function->getNameAsString().c_str());
    Result->AddTextChunk("(");
    AddFunctionParameterChunks(getSema().Context, Function, Result);
    Result->AddTextChunk(")");
    return Result;
  }
  
  if (TemplateDecl *Template = dyn_cast<TemplateDecl>(ND)) {
    CodeCompletionString *Result = new CodeCompletionString;
    Result->AddTextChunk(Template->getNameAsString().c_str());
    Result->AddTextChunk("<");
    AddTemplateParameterChunks(getSema().Context, Template, Result);
    Result->AddTextChunk(">");
    return Result;
  }
  
  return 0;
}

void 
PrintingCodeCompleteConsumer::ProcessCodeCompleteResults(Result *Results, 
                                                         unsigned NumResults) {
  // Sort the results by rank/kind/etc.
  std::stable_sort(Results, Results + NumResults, SortCodeCompleteResult());
  
  // Print the results.
  for (unsigned I = 0; I != NumResults; ++I) {
    switch (Results[I].Kind) {
    case Result::RK_Declaration:
      OS << Results[I].Declaration->getNameAsString() << " : " 
         << Results[I].Rank;
      if (Results[I].Hidden)
        OS << " (Hidden)";
      if (CodeCompletionString *CCS = CreateCodeCompletionString(Results[I])) {
        OS << " : " << CCS->getAsString();
        delete CCS;
      }
        
      OS << '\n';
      break;
      
    case Result::RK_Keyword:
      OS << Results[I].Keyword << " : " << Results[I].Rank << '\n';
      break;
    }
  }
  
  // Once we've printed the code-completion results, suppress remaining
  // diagnostics.
  // FIXME: Move this somewhere else!
  getSema().PP.getDiagnostics().setSuppressAllDiagnostics();
}
