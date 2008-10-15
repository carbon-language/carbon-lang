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
#include "llvm/ADT/DenseMap.h"
using namespace clang;

//===----------------------------------------------------------------------===//
//  Statistics
//===----------------------------------------------------------------------===//

// temporary statistics gathering
static unsigned nFuncs = 0;
static unsigned nVars = 0;
static unsigned nParmVars = 0;
static unsigned nSUC = 0;
static unsigned nCXXSUC = 0;
static unsigned nEnumConst = 0;
static unsigned nEnumDecls = 0;
static unsigned nNamespaces = 0;
static unsigned nTypedef = 0;
static unsigned nFieldDecls = 0;
static unsigned nCXXFieldDecls = 0;
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
  case Typedef:             return "Typedef";
  case Function:            return "Function";
  case Var:                 return "Var";
  case ParmVar:             return "ParmVar";
  case EnumConstant:        return "EnumConstant";
  case ObjCIvar:            return "ObjCIvar";
  case ObjCInterface:       return "ObjCInterface";
  case ObjCClass:           return "ObjCClass";
  case ObjCMethod:          return "ObjCMethod";
  case ObjCProtocol:        return "ObjCProtocol";
  case ObjCForwardProtocol: return "ObjCForwardProtocol"; 
  case Record:              return "Record";
  case CXXRecord:           return "CXXRecord";
  case Enum:                return "Enum";
  case Block:               return "Block";
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
          int(nFuncs+nVars+nParmVars+nFieldDecls+nSUC+nCXXFieldDecls+nCXXSUC+
              nEnumDecls+nEnumConst+nTypedef+nInterfaceDecls+nClassDecls+
              nMethodDecls+nProtocolDecls+nCategoryDecls+nIvarDecls+
              nAtDefsFieldDecls+nNamespaces));
  fprintf(stderr, "    %d namespace decls, %d each (%d bytes)\n", 
          nNamespaces, (int)sizeof(NamespaceDecl), 
          int(nNamespaces*sizeof(NamespaceDecl)));
  fprintf(stderr, "    %d function decls, %d each (%d bytes)\n", 
          nFuncs, (int)sizeof(FunctionDecl), int(nFuncs*sizeof(FunctionDecl)));
  fprintf(stderr, "    %d variable decls, %d each (%d bytes)\n", 
          nVars, (int)sizeof(VarDecl), 
          int(nVars*sizeof(VarDecl)));
  fprintf(stderr, "    %d parameter variable decls, %d each (%d bytes)\n", 
          nParmVars, (int)sizeof(ParmVarDecl),
          int(nParmVars*sizeof(ParmVarDecl)));
  fprintf(stderr, "    %d field decls, %d each (%d bytes)\n", 
          nFieldDecls, (int)sizeof(FieldDecl),
          int(nFieldDecls*sizeof(FieldDecl)));
  fprintf(stderr, "    %d @defs generated field decls, %d each (%d bytes)\n",
          nAtDefsFieldDecls, (int)sizeof(ObjCAtDefsFieldDecl),
          int(nAtDefsFieldDecls*sizeof(ObjCAtDefsFieldDecl)));
  fprintf(stderr, "    %d struct/union/class decls, %d each (%d bytes)\n", 
          nSUC, (int)sizeof(RecordDecl),
          int(nSUC*sizeof(RecordDecl)));
  fprintf(stderr, "    %d C++ field decls, %d each (%d bytes)\n", 
          nCXXFieldDecls, (int)sizeof(CXXFieldDecl),
          int(nCXXFieldDecls*sizeof(CXXFieldDecl)));
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
              nFieldDecls*sizeof(FieldDecl)+nSUC*sizeof(RecordDecl)+
              nCXXFieldDecls*sizeof(CXXFieldDecl)+nCXXSUC*sizeof(CXXRecordDecl)+
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
              nNamespaces*sizeof(NamespaceDecl)));
    
}

void Decl::addDeclKind(Kind k) {
  switch (k) {
  case Namespace:           nNamespaces++; break;
  case Typedef:             nTypedef++; break;
  case Function:            nFuncs++; break;
  case Var:                 nVars++; break;
  case ParmVar:             nParmVars++; break;
  case EnumConstant:        nEnumConst++; break;
  case Field:               nFieldDecls++; break;
  case Record:              nSUC++; break;
  case Enum:                nEnumDecls++; break;
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

  case CXXField:            nCXXFieldDecls++; break;
  case CXXRecord:           nCXXSUC++; break;
  // FIXME: Statistics for C++ decls.
  case CXXMethod:
  case CXXClassVar:
    break;
  }
}

//===----------------------------------------------------------------------===//
// Decl Implementation
//===----------------------------------------------------------------------===//

// Out-of-line virtual method providing a home for Decl.
Decl::~Decl() {
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

  if (ScopedDecl* SD = dyn_cast<ScopedDecl>(this)) {    

    // Observe the unrolled recursion.  By setting N->NextDeclarator = 0x0
    // within the loop, only the Destroy method for the first ScopedDecl
    // will deallocate all of the ScopedDecls in a chain.
    
    ScopedDecl* N = SD->getNextDeclarator();
    
    while (N) {
      ScopedDecl* Tmp = N->getNextDeclarator();
      N->NextDeclarator = 0x0;
      N->Destroy(C);
      N = Tmp;
    }
  }  
  
  this->~Decl();
  C.getAllocator().Deallocate((void *)this);
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

DeclContext *DeclContext::getParent() {
  if (ScopedDecl *SD = dyn_cast<ScopedDecl>(this))
    return SD->getDeclContext();
  else if (BlockDecl *BD = dyn_cast<BlockDecl>(this))
    return BD->getParentContext();
  else
    return NULL;
}
