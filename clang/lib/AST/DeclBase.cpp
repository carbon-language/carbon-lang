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
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <functional>
#include <vector>
using namespace clang;

//===----------------------------------------------------------------------===//
//  Statistics
//===----------------------------------------------------------------------===//

#define DECL(Derived, Base) static int n##Derived##s = 0;
#include "clang/AST/DeclNodes.def"

static bool StatSwitch = false;

// This keeps track of all decl attributes. Since so few decls have attrs, we
// keep them in a hash map instead of wasting space in the Decl class.
typedef llvm::DenseMap<const Decl*, Attr*> DeclAttrMapTy;

static DeclAttrMapTy *DeclAttrs = 0;

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
  if (Enable)
    StatSwitch = true;
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

//===----------------------------------------------------------------------===//
// Decl Implementation
//===----------------------------------------------------------------------===//

void Decl::setDeclContext(DeclContext *DC) {
  if (isOutOfSemaDC())
    delete getMultipleDC();
  
  DeclCtx = reinterpret_cast<uintptr_t>(DC);
}

void Decl::setLexicalDeclContext(DeclContext *DC) {
  if (DC == getLexicalDeclContext())
    return;

  if (isInSemaDC()) {
    MultipleDC *MDC = new MultipleDC();
    MDC->SemanticDC = getDeclContext();
    MDC->LexicalDC = DC;
    DeclCtx = reinterpret_cast<uintptr_t>(MDC) | 0x1;
  } else {
    getMultipleDC()->LexicalDC = DC;
  }
}

// Out-of-line virtual method providing a home for Decl.
Decl::~Decl() {
  if (isOutOfSemaDC())
    delete getMultipleDC();

  if (!HasAttrs)
    return;
  
  DeclAttrMapTy::iterator it = DeclAttrs->find(this);
  assert(it != DeclAttrs->end() && "No attrs found but HasAttrs is true!");

  // release attributes.
  delete it->second;
  invalidateAttrs();
}

void Decl::addAttr(Attr *NewAttr) {
  if (!DeclAttrs)
    DeclAttrs = new DeclAttrMapTy();
  
  Attr *&ExistingAttr = (*DeclAttrs)[this];

  NewAttr->setNext(ExistingAttr);
  ExistingAttr = NewAttr;
  
  HasAttrs = true;
}

void Decl::invalidateAttrs() {
  if (!HasAttrs) return;

  HasAttrs = false;
  (*DeclAttrs)[this] = 0;
  DeclAttrs->erase(this);

  if (DeclAttrs->empty()) {
    delete DeclAttrs;
    DeclAttrs = 0;
  }
}

const Attr *Decl::getAttrs() const {
  if (!HasAttrs)
    return 0;
  
  return (*DeclAttrs)[this];
}

void Decl::swapAttrs(Decl *RHS) {
  bool HasLHSAttr = this->HasAttrs;
  bool HasRHSAttr = RHS->HasAttrs;
  
  // Usually, neither decl has attrs, nothing to do.
  if (!HasLHSAttr && !HasRHSAttr) return;
  
  // If 'this' has no attrs, swap the other way.
  if (!HasLHSAttr)
    return RHS->swapAttrs(this);
  
  // Handle the case when both decls have attrs.
  if (HasRHSAttr) {
    std::swap((*DeclAttrs)[this], (*DeclAttrs)[RHS]);
    return;
  }
  
  // Otherwise, LHS has an attr and RHS doesn't.
  (*DeclAttrs)[RHS] = (*DeclAttrs)[this];
  (*DeclAttrs).erase(this);
  this->HasAttrs = false;
  RHS->HasAttrs = true;
}


void Decl::Destroy(ASTContext& C) {
#if 0
  // FIXME: Once ownership is fully understood, we can enable this code
  if (DeclContext *DC = dyn_cast<DeclContext>(this))
    DC->decls_begin()->Destroy(C);

  // Observe the unrolled recursion.  By setting N->NextDeclInScope = 0x0
  // within the loop, only the Destroy method for the first Decl
  // will deallocate all of the Decls in a chain.
  
  Decl* N = NextDeclInScope;
  
  while (N) {
    Decl* Tmp = N->NextDeclInScope;
    N->NextDeclInScope = 0;
    N->Destroy(C);
    N = Tmp;
  }  

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

// FIXME: We really want to use a DenseSet here to eliminate the
// redundant storage of the declaration names, but (1) it doesn't give
// us the ability to search based on DeclarationName, (2) we really
// need something more like a DenseMultiSet, and (3) it's
// implemented in terms of DenseMap anyway. However, this data
// structure is really space-inefficient, so we'll have to do
// something.
typedef llvm::DenseMap<DeclarationName, std::vector<NamedDecl*> >
  StoredDeclsMap;

DeclContext::~DeclContext() {
  unsigned Size = LookupPtr.getInt();
  if (Size == LookupIsMap) {
    StoredDeclsMap *Map = static_cast<StoredDeclsMap*>(LookupPtr.getPointer());
    delete Map;
  } else {
    NamedDecl **Array = static_cast<NamedDecl**>(LookupPtr.getPointer());
    delete [] Array;
  }
}

void DeclContext::DestroyDecls(ASTContext &C) {
  for (decl_iterator D = decls_begin(); D != decls_end(); )
    (*D++)->Destroy(C);
}

bool DeclContext::isTransparentContext() const {
  if (DeclKind == Decl::Enum)
    return true; // FIXME: Check for C++0x scoped enums
  else if (DeclKind == Decl::LinkageSpec)
    return true;
  else if (DeclKind == Decl::Record || DeclKind == Decl::CXXRecord)
    return cast<RecordDecl>(this)->isAnonymousStructOrUnion();
  else if (DeclKind == Decl::Namespace)
    return false; // FIXME: Check for C++0x inline namespaces

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
      if (TagType *TagT 
            = cast_or_null<TagType>(cast<TagDecl>(this)->TypeForDecl))
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

void DeclContext::addDecl(Decl *D) {
  assert(D->getLexicalDeclContext() == this && "Decl inserted into wrong lexical context");
  assert(!D->NextDeclInScope && D != LastDecl && 
         "Decl already inserted into a DeclContext");

  if (FirstDecl) {
    LastDecl->NextDeclInScope = D;
    LastDecl = D;
  } else {
    FirstDecl = LastDecl = D;
  }

  if (NamedDecl *ND = dyn_cast<NamedDecl>(D))
    ND->getDeclContext()->makeDeclVisibleInContext(ND);
}

/// buildLookup - Build the lookup data structure with all of the
/// declarations in DCtx (and any other contexts linked to it or
/// transparent contexts nested within it).
void DeclContext::buildLookup(DeclContext *DCtx) {
  for (; DCtx; DCtx = DCtx->getNextContext()) {
    for (decl_iterator D = DCtx->decls_begin(), DEnd = DCtx->decls_end(); 
         D != DEnd; ++D) {
      // Insert this declaration into the lookup structure
      if (NamedDecl *ND = dyn_cast<NamedDecl>(*D))
        makeDeclVisibleInContextImpl(ND);

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

  /// If there is no lookup data structure, build one now by walking
  /// all of the linked DeclContexts (in declaration order!) and
  /// inserting their values.
  if (LookupPtr.getPointer() == 0)
    buildLookup(this);

  if (isLookupMap()) {
    StoredDeclsMap *Map = static_cast<StoredDeclsMap*>(LookupPtr.getPointer());
    StoredDeclsMap::iterator Pos = Map->find(Name);
    if (Pos != Map->end())
      return lookup_result(&Pos->second.front(), 
                           &Pos->second.front() + Pos->second.size());
    return lookup_result(0, 0);
  } 

  // We have a small array. Look into it.
  unsigned Size = LookupPtr.getInt();
  NamedDecl **Array = static_cast<NamedDecl**>(LookupPtr.getPointer());
  for (unsigned Idx = 0; Idx != Size; ++Idx)
    if (Array[Idx]->getDeclName() == Name) {
      unsigned Last = Idx + 1;
      while (Last != Size && Array[Last]->getDeclName() == Name)
        ++Last;
      return lookup_result(&Array[Idx], &Array[Last]);
    }

  return lookup_result(0, 0);
}

DeclContext::lookup_const_result 
DeclContext::lookup(DeclarationName Name) const {
  return const_cast<DeclContext*>(this)->lookup(Name);
}

const DeclContext *DeclContext::getLookupContext() const {
  const DeclContext *Ctx = this;
  // Skip through transparent contexts.
  while (Ctx->isTransparentContext())
    Ctx = Ctx->getParent();
  return Ctx;
}

void DeclContext::makeDeclVisibleInContext(NamedDecl *D) {
  // FIXME: This feels like a hack. Should DeclarationName support
  // template-ids, or is there a better way to keep specializations
  // from being visible?
  if (isa<ClassTemplateSpecializationDecl>(D))
    return;

  DeclContext *PrimaryContext = getPrimaryContext();
  if (PrimaryContext != this) {
    PrimaryContext->makeDeclVisibleInContext(D);
    return;
  }

  // If we already have a lookup data structure, perform the insertion
  // into it. Otherwise, be lazy and don't build that structure until
  // someone asks for it.
  if (LookupPtr.getPointer())
    makeDeclVisibleInContextImpl(D);

  // If we are a transparent context, insert into our parent context,
  // too. This operation is recursive.
  if (isTransparentContext())
    getParent()->makeDeclVisibleInContext(D);
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

  bool MayBeRedeclaration = true;

  if (!isLookupMap()) {
    unsigned Size = LookupPtr.getInt();

    // The lookup data is stored as an array. Search through the array
    // to find the insertion location.
    NamedDecl **Array;
    if (Size == 0) {
      Array = new NamedDecl*[LookupIsMap - 1];
      LookupPtr.setPointer(Array);
    } else {
      Array = static_cast<NamedDecl **>(LookupPtr.getPointer());
    }

    // We always keep declarations of the same name next to each other
    // in the array, so that it is easy to return multiple results
    // from lookup(). 
    unsigned FirstMatch;
    for (FirstMatch = 0; FirstMatch != Size; ++FirstMatch)
      if (Array[FirstMatch]->getDeclName() == D->getDeclName())
        break;

    unsigned InsertPos = FirstMatch;
    if (FirstMatch != Size) {
      // We found another declaration with the same name. First
      // determine whether this is a redeclaration of an existing
      // declaration in this scope, in which case we will replace the
      // existing declaration.
      unsigned LastMatch = FirstMatch;
      for (; LastMatch != Size; ++LastMatch) {
        if (Array[LastMatch]->getDeclName() != D->getDeclName())
          break;

        if (D->declarationReplaces(Array[LastMatch])) {
          // D is a redeclaration of an existing element in the
          // array. Replace that element with D.
          Array[LastMatch] = D;
          return;
        }
      }

      // [FirstMatch, LastMatch) contains the set of declarations that
      // have the same name as this declaration. Determine where the
      // declaration D will be inserted into this range.
      if (D->getKind() == Decl::UsingDirective ||
          D->getIdentifierNamespace() == Decl::IDNS_Tag)
        InsertPos = LastMatch;
      else if (Array[LastMatch-1]->getIdentifierNamespace() == Decl::IDNS_Tag)
        InsertPos = LastMatch - 1;
      else
        InsertPos = LastMatch;
    }
       
    if (Size < LookupIsMap - 1) {
      // The new declaration will fit in the array. Insert the new
      // declaration at the position Match in the array. 
      for (unsigned Idx = Size; Idx > InsertPos; --Idx)
       Array[Idx] = Array[Idx-1];
      
      Array[InsertPos] = D;
      LookupPtr.setInt(Size + 1);
      return;
    }

    // We've reached capacity in this array. Create a map and copy in
    // all of the declarations that were stored in the array.
    StoredDeclsMap *Map = new StoredDeclsMap(16);
    LookupPtr.setPointer(Map);
    LookupPtr.setInt(LookupIsMap);
    for (unsigned Idx = 0; Idx != LookupIsMap - 1; ++Idx) 
      makeDeclVisibleInContextImpl(Array[Idx]);
    delete [] Array;

    // Fall through to perform insertion into the map.
    MayBeRedeclaration = false;
  }

  // Insert this declaration into the map.
  StoredDeclsMap *Map = static_cast<StoredDeclsMap*>(LookupPtr.getPointer());
  StoredDeclsMap::iterator Pos = Map->find(D->getDeclName());
  if (Pos == Map->end()) {
    (*Map)[D->getDeclName()].push_back(D);
    return;
  }
  
  if (MayBeRedeclaration) {
    // Determine if this declaration is actually a redeclaration.
    std::vector<NamedDecl *>::iterator Redecl
      = std::find_if(Pos->second.begin(), Pos->second.end(),
                   std::bind1st(std::mem_fun(&NamedDecl::declarationReplaces),
                                D));
    if (Redecl != Pos->second.end()) {
      *Redecl = D;
      return;
    }
  }

  // Put this declaration into the appropriate slot.
  if (D->getKind() == Decl::UsingDirective ||
      D->getIdentifierNamespace() == Decl::IDNS_Tag
      || Pos->second.empty())
    Pos->second.push_back(D);
  else if (Pos->second.back()->getIdentifierNamespace() == Decl::IDNS_Tag) {
    NamedDecl *TagD = Pos->second.back();
    Pos->second.back() = D;
    Pos->second.push_back(TagD);
  } else
    Pos->second.push_back(D);
}

/// Returns iterator range [First, Last) of UsingDirectiveDecls stored within
/// this context.
DeclContext::udir_iterator_range DeclContext::getUsingDirectives() const {
  lookup_const_result Result = lookup(UsingDirectiveDecl::getName());
  return udir_iterator_range(reinterpret_cast<udir_iterator>(Result.first),
                             reinterpret_cast<udir_iterator>(Result.second));
}

