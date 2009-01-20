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
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"
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

// temporary statistics gathering
static unsigned nFuncs = 0;
static unsigned nVars = 0;
static unsigned nParmVars = 0;
static unsigned nOriginalParmVars = 0;
static unsigned nSUC = 0;
static unsigned nCXXSUC = 0;
static unsigned nEnumConst = 0;
static unsigned nEnumDecls = 0;
static unsigned nNamespaces = 0;
static unsigned nOverFuncs = 0;
static unsigned nTypedef = 0;
static unsigned nFieldDecls = 0;
static unsigned nInterfaceDecls = 0;
static unsigned nClassDecls = 0;
static unsigned nMethodDecls = 0;
static unsigned nProtocolDecls = 0;
static unsigned nForwardProtocolDecls = 0;
static unsigned nCategoryDecls = 0;
static unsigned nIvarDecls = 0;
static unsigned nAtDefsFieldDecls = 0;
static unsigned nObjCImplementationDecls = 0;
static unsigned nObjCCategoryImpl = 0;
static unsigned nObjCCompatibleAlias = 0;
static unsigned nObjCPropertyDecl = 0;
static unsigned nObjCPropertyImplDecl = 0;
static unsigned nLinkageSpecDecl = 0;
static unsigned nFileScopeAsmDecl = 0;
static unsigned nBlockDecls = 0;

static bool StatSwitch = false;

// This keeps track of all decl attributes. Since so few decls have attrs, we
// keep them in a hash map instead of wasting space in the Decl class.
typedef llvm::DenseMap<const Decl*, Attr*> DeclAttrMapTy;

static DeclAttrMapTy *DeclAttrs = 0;

const char *Decl::getDeclKindName() const {
  switch (DeclKind) {
  default: assert(0 && "Unknown decl kind!");
  case Namespace:           return "Namespace";
  case OverloadedFunction:  return "OverloadedFunction";
  case Typedef:             return "Typedef";
  case Function:            return "Function";
  case Var:                 return "Var";
  case ParmVar:             return "ParmVar";
  case OriginalParmVar:     return "OriginalParmVar";
  case EnumConstant:        return "EnumConstant";
  case ObjCIvar:            return "ObjCIvar";
  case ObjCInterface:       return "ObjCInterface";
  case ObjCImplementation:  return "ObjCImplementation";
  case ObjCClass:           return "ObjCClass";
  case ObjCMethod:          return "ObjCMethod";
  case ObjCProtocol:        return "ObjCProtocol";
  case ObjCProperty:        return "ObjCProperty";
  case ObjCPropertyImpl:    return "ObjCPropertyImpl";
  case ObjCForwardProtocol: return "ObjCForwardProtocol"; 
  case Record:              return "Record";
  case CXXRecord:           return "CXXRecord";
  case Enum:                return "Enum";
  case Block:               return "Block";
  case Field:               return "Field";
  }
}

const char *DeclContext::getDeclKindName() const {
  switch (DeclKind) {
  default: assert(0 && "Unknown decl kind!");
  case Decl::TranslationUnit:     return "TranslationUnit";
  case Decl::Namespace:           return "Namespace";
  case Decl::OverloadedFunction:  return "OverloadedFunction";
  case Decl::Typedef:             return "Typedef";
  case Decl::Function:            return "Function";
  case Decl::Var:                 return "Var";
  case Decl::ParmVar:             return "ParmVar";
  case Decl::OriginalParmVar:     return "OriginalParmVar";
  case Decl::EnumConstant:        return "EnumConstant";
  case Decl::ObjCIvar:            return "ObjCIvar";
  case Decl::ObjCInterface:       return "ObjCInterface";
  case Decl::ObjCImplementation:  return "ObjCImplementation";
  case Decl::ObjCClass:           return "ObjCClass";
  case Decl::ObjCMethod:          return "ObjCMethod";
  case Decl::ObjCProtocol:        return "ObjCProtocol";
  case Decl::ObjCProperty:        return "ObjCProperty";
  case Decl::ObjCPropertyImpl:    return "ObjCPropertyImpl";
  case Decl::ObjCForwardProtocol: return "ObjCForwardProtocol"; 
  case Decl::Record:              return "Record";
  case Decl::CXXRecord:           return "CXXRecord";
  case Decl::Enum:                return "Enum";
  case Decl::Block:               return "Block";
  case Decl::Field:               return "Field";
  }
}

bool Decl::CollectingStats(bool Enable) {
  if (Enable)
    StatSwitch = true;
  return StatSwitch;
}

void Decl::PrintStats() {
  fprintf(stderr, "*** Decl Stats:\n");
  fprintf(stderr, "  %d decls total.\n", 
          int(nFuncs+nVars+nParmVars+nOriginalParmVars+nFieldDecls+nSUC+nCXXSUC+
              nEnumDecls+nEnumConst+nTypedef+nInterfaceDecls+nClassDecls+
              nMethodDecls+nProtocolDecls+nCategoryDecls+nIvarDecls+
              nAtDefsFieldDecls+nNamespaces+nOverFuncs));
  fprintf(stderr, "    %d namespace decls, %d each (%d bytes)\n", 
          nNamespaces, (int)sizeof(NamespaceDecl), 
          int(nNamespaces*sizeof(NamespaceDecl)));
  fprintf(stderr, "    %d overloaded function decls, %d each (%d bytes)\n", 
          nOverFuncs, (int)sizeof(OverloadedFunctionDecl), 
          int(nOverFuncs*sizeof(OverloadedFunctionDecl)));
  fprintf(stderr, "    %d function decls, %d each (%d bytes)\n", 
          nFuncs, (int)sizeof(FunctionDecl), int(nFuncs*sizeof(FunctionDecl)));
  fprintf(stderr, "    %d variable decls, %d each (%d bytes)\n", 
          nVars, (int)sizeof(VarDecl), 
          int(nVars*sizeof(VarDecl)));
  fprintf(stderr, "    %d parameter variable decls, %d each (%d bytes)\n", 
          nParmVars, (int)sizeof(ParmVarDecl),
          int(nParmVars*sizeof(ParmVarDecl)));
  fprintf(stderr, "    %d original parameter variable decls, %d each (%d bytes)\n", 
          nOriginalParmVars, (int)sizeof(ParmVarWithOriginalTypeDecl),
          int(nOriginalParmVars*sizeof(ParmVarWithOriginalTypeDecl)));
  fprintf(stderr, "    %d field decls, %d each (%d bytes)\n", 
          nFieldDecls, (int)sizeof(FieldDecl),
          int(nFieldDecls*sizeof(FieldDecl)));
  fprintf(stderr, "    %d @defs generated field decls, %d each (%d bytes)\n",
          nAtDefsFieldDecls, (int)sizeof(ObjCAtDefsFieldDecl),
          int(nAtDefsFieldDecls*sizeof(ObjCAtDefsFieldDecl)));
  fprintf(stderr, "    %d struct/union/class decls, %d each (%d bytes)\n", 
          nSUC, (int)sizeof(RecordDecl),
          int(nSUC*sizeof(RecordDecl)));
  fprintf(stderr, "    %d C++ struct/union/class decls, %d each (%d bytes)\n", 
          nCXXSUC, (int)sizeof(CXXRecordDecl),
          int(nCXXSUC*sizeof(CXXRecordDecl)));
  fprintf(stderr, "    %d enum decls, %d each (%d bytes)\n", 
          nEnumDecls, (int)sizeof(EnumDecl), 
          int(nEnumDecls*sizeof(EnumDecl)));
  fprintf(stderr, "    %d enum constant decls, %d each (%d bytes)\n", 
          nEnumConst, (int)sizeof(EnumConstantDecl),
          int(nEnumConst*sizeof(EnumConstantDecl)));
  fprintf(stderr, "    %d typedef decls, %d each (%d bytes)\n", 
          nTypedef, (int)sizeof(TypedefDecl),int(nTypedef*sizeof(TypedefDecl)));
  // Objective-C decls...
  fprintf(stderr, "    %d interface decls, %d each (%d bytes)\n", 
          nInterfaceDecls, (int)sizeof(ObjCInterfaceDecl),
          int(nInterfaceDecls*sizeof(ObjCInterfaceDecl)));
  fprintf(stderr, "    %d instance variable decls, %d each (%d bytes)\n", 
          nIvarDecls, (int)sizeof(ObjCIvarDecl),
          int(nIvarDecls*sizeof(ObjCIvarDecl)));
  fprintf(stderr, "    %d class decls, %d each (%d bytes)\n", 
          nClassDecls, (int)sizeof(ObjCClassDecl),
          int(nClassDecls*sizeof(ObjCClassDecl)));
  fprintf(stderr, "    %d method decls, %d each (%d bytes)\n", 
          nMethodDecls, (int)sizeof(ObjCMethodDecl),
          int(nMethodDecls*sizeof(ObjCMethodDecl)));
  fprintf(stderr, "    %d protocol decls, %d each (%d bytes)\n", 
          nProtocolDecls, (int)sizeof(ObjCProtocolDecl),
          int(nProtocolDecls*sizeof(ObjCProtocolDecl)));
  fprintf(stderr, "    %d forward protocol decls, %d each (%d bytes)\n", 
          nForwardProtocolDecls, (int)sizeof(ObjCForwardProtocolDecl),
          int(nForwardProtocolDecls*sizeof(ObjCForwardProtocolDecl)));
  fprintf(stderr, "    %d category decls, %d each (%d bytes)\n", 
          nCategoryDecls, (int)sizeof(ObjCCategoryDecl),
          int(nCategoryDecls*sizeof(ObjCCategoryDecl)));

  fprintf(stderr, "    %d class implementation decls, %d each (%d bytes)\n", 
          nObjCImplementationDecls, (int)sizeof(ObjCImplementationDecl),
          int(nObjCImplementationDecls*sizeof(ObjCImplementationDecl)));

  fprintf(stderr, "    %d class implementation decls, %d each (%d bytes)\n", 
          nObjCCategoryImpl, (int)sizeof(ObjCCategoryImplDecl),
          int(nObjCCategoryImpl*sizeof(ObjCCategoryImplDecl)));

  fprintf(stderr, "    %d compatibility alias decls, %d each (%d bytes)\n", 
          nObjCCompatibleAlias, (int)sizeof(ObjCCompatibleAliasDecl),
          int(nObjCCompatibleAlias*sizeof(ObjCCompatibleAliasDecl)));
  
  fprintf(stderr, "    %d property decls, %d each (%d bytes)\n", 
          nObjCPropertyDecl, (int)sizeof(ObjCPropertyDecl),
          int(nObjCPropertyDecl*sizeof(ObjCPropertyDecl)));
  
  fprintf(stderr, "    %d property implementation decls, %d each (%d bytes)\n", 
          nObjCPropertyImplDecl, (int)sizeof(ObjCPropertyImplDecl),
          int(nObjCPropertyImplDecl*sizeof(ObjCPropertyImplDecl)));
  
  fprintf(stderr, "Total bytes = %d\n", 
          int(nFuncs*sizeof(FunctionDecl)+
              nVars*sizeof(VarDecl)+nParmVars*sizeof(ParmVarDecl)+
              nOriginalParmVars*sizeof(ParmVarWithOriginalTypeDecl)+
              nFieldDecls*sizeof(FieldDecl)+nSUC*sizeof(RecordDecl)+
              nCXXSUC*sizeof(CXXRecordDecl)+
              nEnumDecls*sizeof(EnumDecl)+nEnumConst*sizeof(EnumConstantDecl)+
              nTypedef*sizeof(TypedefDecl)+
              nInterfaceDecls*sizeof(ObjCInterfaceDecl)+
              nIvarDecls*sizeof(ObjCIvarDecl)+
              nClassDecls*sizeof(ObjCClassDecl)+
              nMethodDecls*sizeof(ObjCMethodDecl)+
              nProtocolDecls*sizeof(ObjCProtocolDecl)+
              nForwardProtocolDecls*sizeof(ObjCForwardProtocolDecl)+
              nCategoryDecls*sizeof(ObjCCategoryDecl)+
              nObjCImplementationDecls*sizeof(ObjCImplementationDecl)+
              nObjCCategoryImpl*sizeof(ObjCCategoryImplDecl)+
              nObjCCompatibleAlias*sizeof(ObjCCompatibleAliasDecl)+
              nObjCPropertyDecl*sizeof(ObjCPropertyDecl)+
              nObjCPropertyImplDecl*sizeof(ObjCPropertyImplDecl)+
              nLinkageSpecDecl*sizeof(LinkageSpecDecl)+
              nFileScopeAsmDecl*sizeof(FileScopeAsmDecl)+
              nNamespaces*sizeof(NamespaceDecl)+
              nOverFuncs*sizeof(OverloadedFunctionDecl)));
    
}

void Decl::addDeclKind(Kind k) {
  switch (k) {
  case Namespace:           nNamespaces++; break;
  case OverloadedFunction:  nOverFuncs++; break;
  case Typedef:             nTypedef++; break;
  case Function:            nFuncs++; break;
  case Var:                 nVars++; break;
  case ParmVar:             nParmVars++; break;
  case OriginalParmVar:     nOriginalParmVars++; break;
  case EnumConstant:        nEnumConst++; break;
  case Field:               nFieldDecls++; break;
  case Record:              nSUC++; break;
  case Enum:                nEnumDecls++; break;
  case ObjCContainer:       break; // is abstract...no need to account for.
  case ObjCInterface:       nInterfaceDecls++; break;
  case ObjCClass:           nClassDecls++; break;
  case ObjCMethod:          nMethodDecls++; break;
  case ObjCProtocol:        nProtocolDecls++; break;
  case ObjCForwardProtocol: nForwardProtocolDecls++; break;
  case ObjCCategory:        nCategoryDecls++; break;
  case ObjCIvar:            nIvarDecls++; break;
  case ObjCAtDefsField:     nAtDefsFieldDecls++; break;
  case ObjCImplementation:  nObjCImplementationDecls++; break;
  case ObjCCategoryImpl:    nObjCCategoryImpl++; break;
  case ObjCCompatibleAlias: nObjCCompatibleAlias++; break;
  case ObjCProperty:        nObjCPropertyDecl++; break;
  case ObjCPropertyImpl:    nObjCPropertyImplDecl++; break;
  case LinkageSpec:         nLinkageSpecDecl++; break;
  case FileScopeAsm:        nFileScopeAsmDecl++; break;
  case Block:               nBlockDecls++; break;
  case ImplicitParam:
  case TranslationUnit:     break;

  case CXXRecord:           nCXXSUC++; break;
  // FIXME: Statistics for C++ decls.
  case TemplateTypeParm:
  case NonTypeTemplateParm:
  case CXXMethod:
  case CXXConstructor:
  case CXXDestructor:
  case CXXConversion:
  case CXXClassVar:
    break;
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
  C.getAllocator().Deallocate((void *)this);
#endif
}

Decl *Decl::castFromDeclContext (const DeclContext *D) {
  return DeclContext::CastTo<Decl>(D);
}

DeclContext *Decl::castToDeclContext(const Decl *D) {
  return DeclContext::CastTo<DeclContext>(D);
}

//===----------------------------------------------------------------------===//
// DeclContext Implementation
//===----------------------------------------------------------------------===//

const DeclContext *DeclContext::getParent() const {
  if (const Decl *D = dyn_cast<Decl>(this))
    return D->getDeclContext();

  return NULL;
}

const DeclContext *DeclContext::getLexicalParent() const {
  if (const Decl *D = dyn_cast<Decl>(this))
    return D->getLexicalDeclContext();

  return getParent();
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

  case Decl::Enum:
  case Decl::Record:
  case Decl::CXXRecord:
    // If this is a tag type that has a definition or is currently
    // being defined, that definition is our primary context.
    if (TagType *TagT = cast_or_null<TagType>(cast<TagDecl>(this)->TypeForDecl))
      if (TagT->isBeingDefined() || 
          (TagT->getDecl() && TagT->getDecl()->isDefinition()))
        return TagT->getDecl();
    return this;

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
    assert(DeclKind >= Decl::FunctionFirst && DeclKind <= Decl::FunctionLast &&
          "Unknown DeclContext kind");
    return this;
  }
}

DeclContext *DeclContext::getNextContext() {
  switch (DeclKind) {
  case Decl::TranslationUnit:
  case Decl::Enum:
  case Decl::Record:
  case Decl::CXXRecord:
  case Decl::ObjCMethod:
  case Decl::ObjCInterface:
  case Decl::ObjCCategory:
  case Decl::ObjCProtocol:
  case Decl::ObjCImplementation:
  case Decl::ObjCCategoryImpl:
  case Decl::LinkageSpec:
  case Decl::Block:
    // There is only one DeclContext for these entities.
    return 0;

  case Decl::Namespace:
    // Return the next namespace
    return static_cast<NamespaceDecl*>(this)->getNextNamespace();

  default:
    assert(DeclKind >= Decl::FunctionFirst && DeclKind <= Decl::FunctionLast &&
          "Unknown DeclContext kind");
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
      if (D->getIdentifierNamespace() == Decl::IDNS_Tag)
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
  if (Pos != Map->end()) {
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
    if (D->getIdentifierNamespace() == Decl::IDNS_Tag || Pos->second.empty())
      Pos->second.push_back(D);
    else if (Pos->second.back()->getIdentifierNamespace() == Decl::IDNS_Tag) {
      NamedDecl *TagD = Pos->second.back();
      Pos->second.back() = D;
      Pos->second.push_back(TagD);
    } else
      Pos->second.push_back(D);
  } else {
    (*Map)[D->getDeclName()].push_back(D);
  }
}
