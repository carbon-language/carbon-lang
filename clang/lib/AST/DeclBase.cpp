//===--- DeclBase.cpp - Declaration AST Node Implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Decl and DeclContext classes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclBase.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclContextInternals.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdio>
#include <vector>
using namespace clang;

//===----------------------------------------------------------------------===//
//  Statistics
//===----------------------------------------------------------------------===//

#define DECL(Derived, Base) static int n##Derived##s = 0;
#include "clang/AST/DeclNodes.def"

static bool StatSwitch = false;

const char *Decl::getDeclKindName() const {
  switch (DeclKind) {
  default: assert(0 && "Declaration not in DeclNodes.def!");
#define DECL(Derived, Base) case Derived: return #Derived;
#include "clang/AST/DeclNodes.def"
  }
}

const char *DeclContext::getDeclKindName() const {
  switch (DeclKind) {
  default: assert(0 && "Declaration context not in DeclNodes.def!");
#define DECL(Derived, Base) case Decl::Derived: return #Derived;
#include "clang/AST/DeclNodes.def"
  }
}

bool Decl::CollectingStats(bool Enable) {
  if (Enable) StatSwitch = true;
  return StatSwitch;
}

void Decl::PrintStats() {
  fprintf(stderr, "*** Decl Stats:\n");

  int totalDecls = 0;
#define DECL(Derived, Base) totalDecls += n##Derived##s;
#include "clang/AST/DeclNodes.def"
  fprintf(stderr, "  %d decls total.\n", totalDecls);

  int totalBytes = 0;
#define DECL(Derived, Base)                                             \
  if (n##Derived##s > 0) {                                              \
    totalBytes += (int)(n##Derived##s * sizeof(Derived##Decl));         \
    fprintf(stderr, "    %d " #Derived " decls, %d each (%d bytes)\n",  \
            n##Derived##s, (int)sizeof(Derived##Decl),                  \
            (int)(n##Derived##s * sizeof(Derived##Decl)));              \
  }
#include "clang/AST/DeclNodes.def"

  fprintf(stderr, "Total bytes = %d\n", totalBytes);
}

void Decl::addDeclKind(Kind k) {
  switch (k) {
  default: assert(0 && "Declaration not in DeclNodes.def!");
#define DECL(Derived, Base) case Derived: ++n##Derived##s; break;
#include "clang/AST/DeclNodes.def"
  }
}

bool Decl::isTemplateParameterPack() const {
  if (const TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(this))
    return TTP->isParameterPack();

  return false;
}

bool Decl::isFunctionOrFunctionTemplate() const {
  if (const UsingShadowDecl *UD = dyn_cast<UsingShadowDecl>(this))
    return UD->getTargetDecl()->isFunctionOrFunctionTemplate();

  return isa<FunctionDecl>(this) || isa<FunctionTemplateDecl>(this);
}

bool Decl::isDefinedOutsideFunctionOrMethod() const {
  for (const DeclContext *DC = getDeclContext(); 
       DC && !DC->isTranslationUnit(); 
       DC = DC->getParent())
    if (DC->isFunctionOrMethod())
      return false;

  return true;
}


//===----------------------------------------------------------------------===//
// PrettyStackTraceDecl Implementation
//===----------------------------------------------------------------------===//

void PrettyStackTraceDecl::print(llvm::raw_ostream &OS) const {
  SourceLocation TheLoc = Loc;
  if (TheLoc.isInvalid() && TheDecl)
    TheLoc = TheDecl->getLocation();

  if (TheLoc.isValid()) {
    TheLoc.print(OS, SM);
    OS << ": ";
  }

  OS << Message;

  if (const NamedDecl *DN = dyn_cast_or_null<NamedDecl>(TheDecl))
    OS << " '" << DN->getQualifiedNameAsString() << '\'';
  OS << '\n';
}

//===----------------------------------------------------------------------===//
// Decl Implementation
//===----------------------------------------------------------------------===//

// Out-of-line virtual method providing a home for Decl.
Decl::~Decl() {
  assert(!HasAttrs && "attributes should have been freed by Destroy");
}

void Decl::setDeclContext(DeclContext *DC) {
  if (isOutOfSemaDC())
    delete getMultipleDC();

  DeclCtx = DC;
}

void Decl::setLexicalDeclContext(DeclContext *DC) {
  if (DC == getLexicalDeclContext())
    return;

  if (isInSemaDC()) {
    MultipleDC *MDC = new (getASTContext()) MultipleDC();
    MDC->SemanticDC = getDeclContext();
    MDC->LexicalDC = DC;
    DeclCtx = MDC;
  } else {
    getMultipleDC()->LexicalDC = DC;
  }
}

bool Decl::isInAnonymousNamespace() const {
  const DeclContext *DC = getDeclContext();
  do {
    if (const NamespaceDecl *ND = dyn_cast<NamespaceDecl>(DC))
      if (ND->isAnonymousNamespace())
        return true;
  } while ((DC = DC->getParent()));

  return false;
}

TranslationUnitDecl *Decl::getTranslationUnitDecl() {
  if (TranslationUnitDecl *TUD = dyn_cast<TranslationUnitDecl>(this))
    return TUD;

  DeclContext *DC = getDeclContext();
  assert(DC && "This decl is not contained in a translation unit!");

  while (!DC->isTranslationUnit()) {
    DC = DC->getParent();
    assert(DC && "This decl is not contained in a translation unit!");
  }

  return cast<TranslationUnitDecl>(DC);
}

ASTContext &Decl::getASTContext() const {
  return getTranslationUnitDecl()->getASTContext();
}

unsigned Decl::getIdentifierNamespaceForKind(Kind DeclKind) {
  switch (DeclKind) {
    case Function:
    case CXXMethod:
    case CXXConstructor:
    case CXXDestructor:
    case CXXConversion:
    case Typedef:
    case EnumConstant:
    case Var:
    case ImplicitParam:
    case ParmVar:
    case NonTypeTemplateParm:
    case ObjCMethod:
    case ObjCContainer:
    case ObjCInterface:
    case ObjCProperty:
    case ObjCCompatibleAlias:
      return IDNS_Ordinary;

    case UsingShadow:
      return 0; // we'll actually overwrite this later

    case UnresolvedUsingValue:
    case UnresolvedUsingTypename:
      return IDNS_Ordinary | IDNS_Using;

    case Using:
      return IDNS_Using;

    case ObjCProtocol:
      return IDNS_ObjCProtocol;

    case ObjCImplementation:
      return IDNS_ObjCImplementation;

    case ObjCCategory:
    case ObjCCategoryImpl:
      return IDNS_ObjCCategoryName;

    case Field:
    case ObjCAtDefsField:
    case ObjCIvar:
      return IDNS_Member;

    case Record:
    case CXXRecord:
    case Enum:
    case TemplateTypeParm:
      return IDNS_Tag;

    case Namespace:
    case Template:
    case FunctionTemplate:
    case ClassTemplate:
    case TemplateTemplateParm:
    case NamespaceAlias:
      return IDNS_Tag | IDNS_Ordinary;

    // Never have names.
    case Friend:
    case FriendTemplate:
    case LinkageSpec:
    case FileScopeAsm:
    case StaticAssert:
    case ObjCClass:
    case ObjCPropertyImpl:
    case ObjCForwardProtocol:
    case Block:
    case TranslationUnit:

    // Aren't looked up?
    case UsingDirective:
    case ClassTemplateSpecialization:
    case ClassTemplatePartialSpecialization:
      return 0;
  }

  return 0;
}

void Decl::addAttr(Attr *NewAttr) {
  Attr *&ExistingAttr = getASTContext().getDeclAttrs(this);

  NewAttr->setNext(ExistingAttr);
  ExistingAttr = NewAttr;

  HasAttrs = true;
}

void Decl::invalidateAttrs() {
  if (!HasAttrs) return;

  HasAttrs = false;
  getASTContext().eraseDeclAttrs(this);
}

const Attr *Decl::getAttrsImpl() const {
  assert(HasAttrs && "getAttrs() should verify this!");
  return getASTContext().getDeclAttrs(this);
}

void Decl::swapAttrs(Decl *RHS) {
  bool HasLHSAttr = this->HasAttrs;
  bool HasRHSAttr = RHS->HasAttrs;

  // Usually, neither decl has attrs, nothing to do.
  if (!HasLHSAttr && !HasRHSAttr) return;

  // If 'this' has no attrs, swap the other way.
  if (!HasLHSAttr)
    return RHS->swapAttrs(this);

  ASTContext &Context = getASTContext();

  // Handle the case when both decls have attrs.
  if (HasRHSAttr) {
    std::swap(Context.getDeclAttrs(this), Context.getDeclAttrs(RHS));
    return;
  }

  // Otherwise, LHS has an attr and RHS doesn't.
  Context.getDeclAttrs(RHS) = Context.getDeclAttrs(this);
  Context.eraseDeclAttrs(this);
  this->HasAttrs = false;
  RHS->HasAttrs = true;
}


void Decl::Destroy(ASTContext &C) {
  // Free attributes for this decl.
  if (HasAttrs) {
    C.getDeclAttrs(this)->Destroy(C);
    invalidateAttrs();
    HasAttrs = false;
  }

#if 0
  // FIXME: Once ownership is fully understood, we can enable this code
  if (DeclContext *DC = dyn_cast<DeclContext>(this))
    DC->decls_begin()->Destroy(C);

  // Observe the unrolled recursion.  By setting N->NextDeclInContext = 0x0
  // within the loop, only the Destroy method for the first Decl
  // will deallocate all of the Decls in a chain.

  Decl* N = getNextDeclInContext();

  while (N) {
    Decl* Tmp = N->getNextDeclInContext();
    N->NextDeclInContext = 0;
    N->Destroy(C);
    N = Tmp;
  }

  if (isOutOfSemaDC())
    delete (C) getMultipleDC();
  
  this->~Decl();
  C.Deallocate((void *)this);
#endif  
}

Decl *Decl::castFromDeclContext (const DeclContext *D) {
  Decl::Kind DK = D->getDeclKind();
  switch(DK) {
#define DECL_CONTEXT(Name) \
    case Decl::Name:     \
      return static_cast<Name##Decl*>(const_cast<DeclContext*>(D));
#define DECL_CONTEXT_BASE(Name)
#include "clang/AST/DeclNodes.def"
    default:
#define DECL_CONTEXT_BASE(Name)                                   \
      if (DK >= Decl::Name##First && DK <= Decl::Name##Last)    \
        return static_cast<Name##Decl*>(const_cast<DeclContext*>(D));
#include "clang/AST/DeclNodes.def"
      assert(false && "a decl that inherits DeclContext isn't handled");
      return 0;
  }
}

DeclContext *Decl::castToDeclContext(const Decl *D) {
  Decl::Kind DK = D->getKind();
  switch(DK) {
#define DECL_CONTEXT(Name) \
    case Decl::Name:     \
      return static_cast<Name##Decl*>(const_cast<Decl*>(D));
#define DECL_CONTEXT_BASE(Name)
#include "clang/AST/DeclNodes.def"
    default:
#define DECL_CONTEXT_BASE(Name)                                   \
      if (DK >= Decl::Name##First && DK <= Decl::Name##Last)    \
        return static_cast<Name##Decl*>(const_cast<Decl*>(D));
#include "clang/AST/DeclNodes.def"
      assert(false && "a decl that inherits DeclContext isn't handled");
      return 0;
  }
}

CompoundStmt* Decl::getCompoundBody() const {
  return dyn_cast_or_null<CompoundStmt>(getBody());
}

SourceLocation Decl::getBodyRBrace() const {
  Stmt *Body = getBody();
  if (!Body)
    return SourceLocation();
  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Body))
    return CS->getRBracLoc();
  assert(isa<CXXTryStmt>(Body) &&
         "Body can only be CompoundStmt or CXXTryStmt");
  return cast<CXXTryStmt>(Body)->getSourceRange().getEnd();
}

#ifndef NDEBUG
void Decl::CheckAccessDeclContext() const {
  // Suppress this check if any of the following hold:
  // 1. this is the translation unit (and thus has no parent)
  // 2. this is a template parameter (and thus doesn't belong to its context)
  // 3. this is a ParmVarDecl (which can be in a record context during
  //    the brief period between its creation and the creation of the
  //    FunctionDecl)
  // 4. the context is not a record
  if (isa<TranslationUnitDecl>(this) ||
      isTemplateParameter() ||
      isa<ParmVarDecl>(this) ||
      !isa<CXXRecordDecl>(getDeclContext()))
    return;

  assert(Access != AS_none &&
         "Access specifier is AS_none inside a record decl");
}

#endif

//===----------------------------------------------------------------------===//
// DeclContext Implementation
//===----------------------------------------------------------------------===//

bool DeclContext::classof(const Decl *D) {
  switch (D->getKind()) {
#define DECL_CONTEXT(Name) case Decl::Name:
#define DECL_CONTEXT_BASE(Name)
#include "clang/AST/DeclNodes.def"
      return true;
    default:
#define DECL_CONTEXT_BASE(Name)                   \
      if (D->getKind() >= Decl::Name##First &&  \
          D->getKind() <= Decl::Name##Last)     \
        return true;
#include "clang/AST/DeclNodes.def"
      return false;
  }
}

DeclContext::~DeclContext() {
  delete static_cast<StoredDeclsMap*>(LookupPtr);
}

void DeclContext::DestroyDecls(ASTContext &C) {
  for (decl_iterator D = decls_begin(); D != decls_end(); )
    (*D++)->Destroy(C);
}

/// \brief Find the parent context of this context that will be
/// used for unqualified name lookup.
///
/// Generally, the parent lookup context is the semantic context. However, for
/// a friend function the parent lookup context is the lexical context, which
/// is the class in which the friend is declared.
DeclContext *DeclContext::getLookupParent() {
  // FIXME: Find a better way to identify friends
  if (isa<FunctionDecl>(this))
    if (getParent()->getLookupContext()->isFileContext() &&
        getLexicalParent()->getLookupContext()->isRecord())
      return getLexicalParent();
  
  return getParent();
}

bool DeclContext::isDependentContext() const {
  if (isFileContext())
    return false;

  if (isa<ClassTemplatePartialSpecializationDecl>(this))
    return true;

  if (const CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(this))
    if (Record->getDescribedClassTemplate())
      return true;

  if (const FunctionDecl *Function = dyn_cast<FunctionDecl>(this))
    if (Function->getDescribedFunctionTemplate())
      return true;

  return getParent() && getParent()->isDependentContext();
}

bool DeclContext::isTransparentContext() const {
  if (DeclKind == Decl::Enum)
    return true; // FIXME: Check for C++0x scoped enums
  else if (DeclKind == Decl::LinkageSpec)
    return true;
  else if (DeclKind >= Decl::RecordFirst && DeclKind <= Decl::RecordLast)
    return cast<RecordDecl>(this)->isAnonymousStructOrUnion();
  else if (DeclKind == Decl::Namespace)
    return false; // FIXME: Check for C++0x inline namespaces

  return false;
}

bool DeclContext::Encloses(DeclContext *DC) {
  if (getPrimaryContext() != this)
    return getPrimaryContext()->Encloses(DC);

  for (; DC; DC = DC->getParent())
    if (DC->getPrimaryContext() == this)
      return true;
  return false;
}

DeclContext *DeclContext::getPrimaryContext() {
  switch (DeclKind) {
  case Decl::TranslationUnit:
  case Decl::LinkageSpec:
  case Decl::Block:
    // There is only one DeclContext for these entities.
    return this;

  case Decl::Namespace:
    // The original namespace is our primary context.
    return static_cast<NamespaceDecl*>(this)->getOriginalNamespace();

  case Decl::ObjCMethod:
    return this;

  case Decl::ObjCInterface:
  case Decl::ObjCProtocol:
  case Decl::ObjCCategory:
    // FIXME: Can Objective-C interfaces be forward-declared?
    return this;

  case Decl::ObjCImplementation:
  case Decl::ObjCCategoryImpl:
    return this;

  default:
    if (DeclKind >= Decl::TagFirst && DeclKind <= Decl::TagLast) {
      // If this is a tag type that has a definition or is currently
      // being defined, that definition is our primary context.
      if (const TagType *TagT =cast<TagDecl>(this)->TypeForDecl->getAs<TagType>())
        if (TagT->isBeingDefined() ||
            (TagT->getDecl() && TagT->getDecl()->isDefinition()))
          return TagT->getDecl();
      return this;
    }

    assert(DeclKind >= Decl::FunctionFirst && DeclKind <= Decl::FunctionLast &&
          "Unknown DeclContext kind");
    return this;
  }
}

DeclContext *DeclContext::getNextContext() {
  switch (DeclKind) {
  case Decl::Namespace:
    // Return the next namespace
    return static_cast<NamespaceDecl*>(this)->getNextNamespace();

  default:
    return 0;
  }
}

/// \brief Load the declarations within this lexical storage from an
/// external source.
void
DeclContext::LoadLexicalDeclsFromExternalStorage() const {
  ExternalASTSource *Source = getParentASTContext().getExternalSource();
  assert(hasExternalLexicalStorage() && Source && "No external storage?");

  llvm::SmallVector<uint32_t, 64> Decls;
  if (Source->ReadDeclsLexicallyInContext(const_cast<DeclContext *>(this),
                                          Decls))
    return;

  // There is no longer any lexical storage in this context
  ExternalLexicalStorage = false;

  if (Decls.empty())
    return;

  // Resolve all of the declaration IDs into declarations, building up
  // a chain of declarations via the Decl::NextDeclInContext field.
  Decl *FirstNewDecl = 0;
  Decl *PrevDecl = 0;
  for (unsigned I = 0, N = Decls.size(); I != N; ++I) {
    Decl *D = Source->GetDecl(Decls[I]);
    if (PrevDecl)
      PrevDecl->NextDeclInContext = D;
    else
      FirstNewDecl = D;

    PrevDecl = D;
  }

  // Splice the newly-read declarations into the beginning of the list
  // of declarations.
  PrevDecl->NextDeclInContext = FirstDecl;
  FirstDecl = FirstNewDecl;
  if (!LastDecl)
    LastDecl = PrevDecl;
}

void
DeclContext::LoadVisibleDeclsFromExternalStorage() const {
  DeclContext *This = const_cast<DeclContext *>(this);
  ExternalASTSource *Source = getParentASTContext().getExternalSource();
  assert(hasExternalVisibleStorage() && Source && "No external storage?");

  llvm::SmallVector<VisibleDeclaration, 64> Decls;
  if (Source->ReadDeclsVisibleInContext(This, Decls))
    return;

  // There is no longer any visible storage in this context
  ExternalVisibleStorage = false;

  // Load the declaration IDs for all of the names visible in this
  // context.
  assert(!LookupPtr && "Have a lookup map before de-serialization?");
  StoredDeclsMap *Map = new StoredDeclsMap;
  LookupPtr = Map;
  for (unsigned I = 0, N = Decls.size(); I != N; ++I) {
    (*Map)[Decls[I].Name].setFromDeclIDs(Decls[I].Declarations);
  }
}

DeclContext::decl_iterator DeclContext::decls_begin() const {
  if (hasExternalLexicalStorage())
    LoadLexicalDeclsFromExternalStorage();

  // FIXME: Check whether we need to load some declarations from
  // external storage.
  return decl_iterator(FirstDecl);
}

DeclContext::decl_iterator DeclContext::decls_end() const {
  if (hasExternalLexicalStorage())
    LoadLexicalDeclsFromExternalStorage();

  return decl_iterator();
}

bool DeclContext::decls_empty() const {
  if (hasExternalLexicalStorage())
    LoadLexicalDeclsFromExternalStorage();

  return !FirstDecl;
}

void DeclContext::removeDecl(Decl *D) {
  assert(D->getLexicalDeclContext() == this &&
         "decl being removed from non-lexical context");
  assert((D->NextDeclInContext || D == LastDecl) &&
         "decl is not in decls list");

  // Remove D from the decl chain.  This is O(n) but hopefully rare.
  if (D == FirstDecl) {
    if (D == LastDecl)
      FirstDecl = LastDecl = 0;
    else
      FirstDecl = D->NextDeclInContext;
  } else {
    for (Decl *I = FirstDecl; true; I = I->NextDeclInContext) {
      assert(I && "decl not found in linked list");
      if (I->NextDeclInContext == D) {
        I->NextDeclInContext = D->NextDeclInContext;
        if (D == LastDecl) LastDecl = I;
        break;
      }
    }
  }
  
  // Mark that D is no longer in the decl chain.
  D->NextDeclInContext = 0;

  // Remove D from the lookup table if necessary.
  if (isa<NamedDecl>(D)) {
    NamedDecl *ND = cast<NamedDecl>(D);

    void *OpaqueMap = getPrimaryContext()->LookupPtr;
    if (!OpaqueMap) return;

    StoredDeclsMap *Map = static_cast<StoredDeclsMap*>(OpaqueMap);
    StoredDeclsMap::iterator Pos = Map->find(ND->getDeclName());
    assert(Pos != Map->end() && "no lookup entry for decl");
    Pos->second.remove(ND);
  }
}

void DeclContext::addHiddenDecl(Decl *D) {
  assert(D->getLexicalDeclContext() == this &&
         "Decl inserted into wrong lexical context");
  assert(!D->getNextDeclInContext() && D != LastDecl &&
         "Decl already inserted into a DeclContext");

  if (FirstDecl) {
    LastDecl->NextDeclInContext = D;
    LastDecl = D;
  } else {
    FirstDecl = LastDecl = D;
  }
}

void DeclContext::addDecl(Decl *D) {
  addHiddenDecl(D);

  if (NamedDecl *ND = dyn_cast<NamedDecl>(D))
    ND->getDeclContext()->makeDeclVisibleInContext(ND);
}

/// buildLookup - Build the lookup data structure with all of the
/// declarations in DCtx (and any other contexts linked to it or
/// transparent contexts nested within it).
void DeclContext::buildLookup(DeclContext *DCtx) {
  for (; DCtx; DCtx = DCtx->getNextContext()) {
    for (decl_iterator D = DCtx->decls_begin(),
                    DEnd = DCtx->decls_end();
         D != DEnd; ++D) {
      // Insert this declaration into the lookup structure, but only
      // if it's semantically in its decl context.  During non-lazy
      // lookup building, this is implicitly enforced by addDecl.
      if (NamedDecl *ND = dyn_cast<NamedDecl>(*D))
        if (D->getDeclContext() == DCtx)
          makeDeclVisibleInContextImpl(ND);

      // Insert any forward-declared Objective-C interfaces into the lookup
      // data structure.
      if (ObjCClassDecl *Class = dyn_cast<ObjCClassDecl>(*D))
        for (ObjCClassDecl::iterator I = Class->begin(), IEnd = Class->end();
             I != IEnd; ++I)
          makeDeclVisibleInContextImpl(I->getInterface());
      
      // If this declaration is itself a transparent declaration context,
      // add its members (recursively).
      if (DeclContext *InnerCtx = dyn_cast<DeclContext>(*D))
        if (InnerCtx->isTransparentContext())
          buildLookup(InnerCtx->getPrimaryContext());
    }
  }
}

DeclContext::lookup_result
DeclContext::lookup(DeclarationName Name) {
  DeclContext *PrimaryContext = getPrimaryContext();
  if (PrimaryContext != this)
    return PrimaryContext->lookup(Name);

  if (hasExternalVisibleStorage())
    LoadVisibleDeclsFromExternalStorage();

  /// If there is no lookup data structure, build one now by walking
  /// all of the linked DeclContexts (in declaration order!) and
  /// inserting their values.
  if (!LookupPtr) {
    buildLookup(this);

    if (!LookupPtr)
      return lookup_result(0, 0);
  }

  StoredDeclsMap *Map = static_cast<StoredDeclsMap*>(LookupPtr);
  StoredDeclsMap::iterator Pos = Map->find(Name);
  if (Pos == Map->end())
    return lookup_result(0, 0);
  return Pos->second.getLookupResult(getParentASTContext());
}

DeclContext::lookup_const_result
DeclContext::lookup(DeclarationName Name) const {
  return const_cast<DeclContext*>(this)->lookup(Name);
}

DeclContext *DeclContext::getLookupContext() {
  DeclContext *Ctx = this;
  // Skip through transparent contexts.
  while (Ctx->isTransparentContext())
    Ctx = Ctx->getParent();
  return Ctx;
}

DeclContext *DeclContext::getEnclosingNamespaceContext() {
  DeclContext *Ctx = this;
  // Skip through non-namespace, non-translation-unit contexts.
  while (!Ctx->isFileContext() || Ctx->isTransparentContext())
    Ctx = Ctx->getParent();
  return Ctx->getPrimaryContext();
}

void DeclContext::makeDeclVisibleInContext(NamedDecl *D, bool Recoverable) {
  // FIXME: This feels like a hack. Should DeclarationName support
  // template-ids, or is there a better way to keep specializations
  // from being visible?
  if (isa<ClassTemplateSpecializationDecl>(D))
    return;
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    if (FD->isFunctionTemplateSpecialization())
      return;

  DeclContext *PrimaryContext = getPrimaryContext();
  if (PrimaryContext != this) {
    PrimaryContext->makeDeclVisibleInContext(D, Recoverable);
    return;
  }

  // If we already have a lookup data structure, perform the insertion
  // into it. Otherwise, be lazy and don't build that structure until
  // someone asks for it.
  if (LookupPtr || !Recoverable)
    makeDeclVisibleInContextImpl(D);

  // If we are a transparent context, insert into our parent context,
  // too. This operation is recursive.
  if (isTransparentContext())
    getParent()->makeDeclVisibleInContext(D, Recoverable);
}

void DeclContext::makeDeclVisibleInContextImpl(NamedDecl *D) {
  // Skip unnamed declarations.
  if (!D->getDeclName())
    return;

  // FIXME: This feels like a hack. Should DeclarationName support
  // template-ids, or is there a better way to keep specializations
  // from being visible?
  if (isa<ClassTemplateSpecializationDecl>(D))
    return;

  if (!LookupPtr)
    LookupPtr = new StoredDeclsMap;

  // Insert this declaration into the map.
  StoredDeclsMap &Map = *static_cast<StoredDeclsMap*>(LookupPtr);
  StoredDeclsList &DeclNameEntries = Map[D->getDeclName()];
  if (DeclNameEntries.isNull()) {
    DeclNameEntries.setOnlyValue(D);
    return;
  }

  // If it is possible that this is a redeclaration, check to see if there is
  // already a decl for which declarationReplaces returns true.  If there is
  // one, just replace it and return.
  if (DeclNameEntries.HandleRedeclaration(getParentASTContext(), D))
    return;

  // Put this declaration into the appropriate slot.
  DeclNameEntries.AddSubsequentDecl(D);
}

/// Returns iterator range [First, Last) of UsingDirectiveDecls stored within
/// this context.
DeclContext::udir_iterator_range
DeclContext::getUsingDirectives() const {
  lookup_const_result Result = lookup(UsingDirectiveDecl::getName());
  return udir_iterator_range(reinterpret_cast<udir_iterator>(Result.first),
                             reinterpret_cast<udir_iterator>(Result.second));
}

void StoredDeclsList::materializeDecls(ASTContext &Context) {
  if (isNull())
    return;

  switch ((DataKind)(Data & 0x03)) {
  case DK_Decl:
  case DK_Decl_Vector:
    break;

  case DK_DeclID: {
    // Resolve this declaration ID to an actual declaration by
    // querying the external AST source.
    unsigned DeclID = Data >> 2;

    ExternalASTSource *Source = Context.getExternalSource();
    assert(Source && "No external AST source available!");

    Data = reinterpret_cast<uintptr_t>(Source->GetDecl(DeclID));
    break;
  }

  case DK_ID_Vector: {
    // We have a vector of declaration IDs. Resolve all of them to
    // actual declarations.
    VectorTy &Vector = *getAsVector();
    ExternalASTSource *Source = Context.getExternalSource();
    assert(Source && "No external AST source available!");

    for (unsigned I = 0, N = Vector.size(); I != N; ++I)
      Vector[I] = reinterpret_cast<uintptr_t>(Source->GetDecl(Vector[I]));

    Data = (Data & ~0x03) | DK_Decl_Vector;
    break;
  }
  }
}
