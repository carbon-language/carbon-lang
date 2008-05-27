//===--- Decl.cpp - Declaration AST Node Implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Decl class and subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/Basic/IdentifierTable.h"
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
static unsigned nEnumConst = 0;
static unsigned nEnumDecls = 0;
static unsigned nNamespaces = 0;
static unsigned nTypedef = 0;
static unsigned nFieldDecls = 0;
static unsigned nInterfaceDecls = 0;
static unsigned nClassDecls = 0;
static unsigned nMethodDecls = 0;
static unsigned nProtocolDecls = 0;
static unsigned nForwardProtocolDecls = 0;
static unsigned nCategoryDecls = 0;
static unsigned nIvarDecls = 0;
static unsigned nObjCImplementationDecls = 0;
static unsigned nObjCCategoryImpl = 0;
static unsigned nObjCCompatibleAlias = 0;
static unsigned nObjCPropertyDecl = 0;
static unsigned nObjCPropertyImplDecl = 0;
static unsigned nLinkageSpecDecl = 0;
static unsigned nFileScopeAsmDecl = 0;

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
  case Struct:              return "Struct";
  case Union:               return "Union";
  case Class:               return "Class";
  case Enum:                return "Enum";
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
          int(nFuncs+nVars+nParmVars+nFieldDecls+nSUC+
              nEnumDecls+nEnumConst+nTypedef+nInterfaceDecls+nClassDecls+
              nMethodDecls+nProtocolDecls+nCategoryDecls+nIvarDecls+
              nNamespaces));
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
  fprintf(stderr, "    %d struct/union/class decls, %d each (%d bytes)\n", 
          nSUC, (int)sizeof(RecordDecl),
          int(nSUC*sizeof(RecordDecl)));
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
  case Struct: case Union: case Class: nSUC++; break;
  case Enum:                nEnumDecls++; break;
  case ObjCInterface:       nInterfaceDecls++; break;
  case ObjCClass:           nClassDecls++; break;
  case ObjCMethod:          nMethodDecls++; break;
  case ObjCProtocol:        nProtocolDecls++; break;
  case ObjCForwardProtocol: nForwardProtocolDecls++; break;
  case ObjCCategory:        nCategoryDecls++; break;
  case ObjCIvar:            nIvarDecls++; break;
  case ObjCImplementation:  nObjCImplementationDecls++; break;
  case ObjCCategoryImpl:    nObjCCategoryImpl++; break;
  case ObjCCompatibleAlias: nObjCCompatibleAlias++; break;
  case ObjCProperty:        nObjCPropertyDecl++; break;
  case ObjCPropertyImpl:    nObjCPropertyImplDecl++; break;
  case LinkageSpec:         nLinkageSpecDecl++; break;
  case FileScopeAsm:        nFileScopeAsmDecl++; break;
  case TranslationUnit:     break;
  }
}

//===----------------------------------------------------------------------===//
// Decl Allocation/Deallocation Method Implementations
//===----------------------------------------------------------------------===//
 
TranslationUnitDecl *TranslationUnitDecl::Create(ASTContext &C) {
  void *Mem = C.getAllocator().Allocate<TranslationUnitDecl>();
  return new (Mem) TranslationUnitDecl();
}

NamespaceDecl *NamespaceDecl::Create(ASTContext &C, DeclContext *DC,
                                     SourceLocation L, IdentifierInfo *Id) {
  void *Mem = C.getAllocator().Allocate<NamespaceDecl>();
  return new (Mem) NamespaceDecl(DC, L, Id);
}

void NamespaceDecl::Destroy(ASTContext& C) {
  // NamespaceDecl uses "NextDeclarator" to chain namespace declarations
  // together. They are all top-level Decls.
  
  this->~NamespaceDecl();
  C.getAllocator().Deallocate((void *)this);
}


VarDecl *VarDecl::Create(ASTContext &C, DeclContext *DC,
                         SourceLocation L,
                         IdentifierInfo *Id, QualType T,
                         StorageClass S, ScopedDecl *PrevDecl) {
  void *Mem = C.getAllocator().Allocate<VarDecl>();
  return new (Mem) VarDecl(Var, DC, L, Id, T, S, PrevDecl);
}

ParmVarDecl *ParmVarDecl::Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation L, IdentifierInfo *Id,
                                 QualType T, StorageClass S,
                                 Expr *DefArg, ScopedDecl *PrevDecl) {
  void *Mem = C.getAllocator().Allocate<ParmVarDecl>();
  return new (Mem) ParmVarDecl(DC, L, Id, T, S, DefArg, PrevDecl);
}

FunctionDecl *FunctionDecl::Create(ASTContext &C, DeclContext *DC,
                                   SourceLocation L, 
                                   IdentifierInfo *Id, QualType T, 
                                   StorageClass S, bool isInline, 
                                   ScopedDecl *PrevDecl) {
  void *Mem = C.getAllocator().Allocate<FunctionDecl>();
  return new (Mem) FunctionDecl(DC, L, Id, T, S, isInline, PrevDecl);
}

FieldDecl *FieldDecl::Create(ASTContext &C, SourceLocation L,
                             IdentifierInfo *Id, QualType T, Expr *BW) {
  void *Mem = C.getAllocator().Allocate<FieldDecl>();
  return new (Mem) FieldDecl(L, Id, T, BW);
}


EnumConstantDecl *EnumConstantDecl::Create(ASTContext &C, EnumDecl *CD,
                                           SourceLocation L,
                                           IdentifierInfo *Id, QualType T,
                                           Expr *E, const llvm::APSInt &V, 
                                           ScopedDecl *PrevDecl){
  void *Mem = C.getAllocator().Allocate<EnumConstantDecl>();
  return new (Mem) EnumConstantDecl(CD, L, Id, T, E, V, PrevDecl);
}

void EnumConstantDecl::Destroy(ASTContext& C) {
  if (Init) Init->Destroy(C);
  Decl::Destroy(C);
}

TypedefDecl *TypedefDecl::Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation L,
                                 IdentifierInfo *Id, QualType T,
                                 ScopedDecl *PD) {
  void *Mem = C.getAllocator().Allocate<TypedefDecl>();
  return new (Mem) TypedefDecl(DC, L, Id, T, PD);
}

EnumDecl *EnumDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                           IdentifierInfo *Id,
                           ScopedDecl *PrevDecl) {
  void *Mem = C.getAllocator().Allocate<EnumDecl>();
  return new (Mem) EnumDecl(DC, L, Id, PrevDecl);
}

RecordDecl *RecordDecl::Create(ASTContext &C, Kind DK, DeclContext *DC,
                               SourceLocation L, IdentifierInfo *Id,
                               ScopedDecl *PrevDecl) {
  void *Mem = C.getAllocator().Allocate<RecordDecl>();
  return new (Mem) RecordDecl(DK, DC, L, Id, PrevDecl);
}

void EnumDecl::Destroy(ASTContext& C) {
  if (ElementList) ElementList->Destroy(C);
  Decl::Destroy(C);
}


FileScopeAsmDecl *FileScopeAsmDecl::Create(ASTContext &C,
                                           SourceLocation L,
                                           StringLiteral *Str) {
  void *Mem = C.getAllocator().Allocate<FileScopeAsmDecl>();
  return new (Mem) FileScopeAsmDecl(L, Str);
}

LinkageSpecDecl *LinkageSpecDecl::Create(ASTContext &C,
                                         SourceLocation L,
                                         LanguageIDs Lang, Decl *D) {
  void *Mem = C.getAllocator().Allocate<LinkageSpecDecl>();
  return new (Mem) LinkageSpecDecl(L, Lang, D);
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

  // FIXME: Properly release attributes.
  // delete it->second;
  DeclAttrs->erase(it);
  
  if (DeclAttrs->empty()) {
    delete DeclAttrs;
    DeclAttrs = 0;
  }        
}

void Decl::addAttr(Attr *NewAttr) {
  if (!DeclAttrs)
    DeclAttrs = new DeclAttrMapTy();
  
  Attr *&ExistingAttr = (*DeclAttrs)[this];

  NewAttr->setNext(ExistingAttr);
  ExistingAttr = NewAttr;
  
  HasAttrs = true;
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

//===----------------------------------------------------------------------===//
// DeclContext Implementation
//===----------------------------------------------------------------------===//

DeclContext *DeclContext::getParent() const {
  if (ScopedDecl *SD = dyn_cast<ScopedDecl>(this))
    return SD->getDeclContext();
  else
    return NULL;
}

Decl *DeclContext::ToDecl (const DeclContext *D) {
  return CastTo<Decl>(D);
}

DeclContext *DeclContext::FromDecl (const Decl *D) {
  return CastTo<DeclContext>(D);
}

//===----------------------------------------------------------------------===//
// NamedDecl Implementation
//===----------------------------------------------------------------------===//

const char *NamedDecl::getName() const {
  if (const IdentifierInfo *II = getIdentifier())
    return II->getName();
  return "";
}

//===----------------------------------------------------------------------===//
// FunctionDecl Implementation
//===----------------------------------------------------------------------===//

FunctionDecl::~FunctionDecl() {
  delete[] ParamInfo;
}

void FunctionDecl::Destroy(ASTContext& C) {
  if (Body)
    Body->Destroy(C);

  for (param_iterator I=param_begin(), E=param_end(); I!=E; ++I)
    (*I)->Destroy(C);
    
  Decl::Destroy(C);
}


Stmt *FunctionDecl::getBody(const FunctionDecl *&Definition) const {
  for (const FunctionDecl *FD = this; FD != 0; FD = FD->PreviousDeclaration) {
    if (FD->Body) {
      Definition = FD;
      return FD->Body;
    }
  }

  return 0;
}

unsigned FunctionDecl::getNumParams() const {
  const FunctionType *FT = getType()->getAsFunctionType();
  if (isa<FunctionTypeNoProto>(FT))
    return 0;
  return cast<FunctionTypeProto>(FT)->getNumArgs();
}

void FunctionDecl::setParams(ParmVarDecl **NewParamInfo, unsigned NumParams) {
  assert(ParamInfo == 0 && "Already has param info!");
  assert(NumParams == getNumParams() && "Parameter count mismatch!");
  
  // Zero params -> null pointer.
  if (NumParams) {
    ParamInfo = new ParmVarDecl*[NumParams];
    memcpy(ParamInfo, NewParamInfo, sizeof(ParmVarDecl*)*NumParams);
  }
}

/// getMinRequiredArguments - Returns the minimum number of arguments
/// needed to call this function. This may be fewer than the number of
/// function parameters, if some of the parameters have default
/// arguments (in C++).
unsigned FunctionDecl::getMinRequiredArguments() const {
  unsigned NumRequiredArgs = getNumParams();
  while (NumRequiredArgs > 0
         && getParamDecl(NumRequiredArgs-1)->getDefaultArg())
    --NumRequiredArgs;

  return NumRequiredArgs;
}

//===----------------------------------------------------------------------===//
// RecordDecl Implementation
//===----------------------------------------------------------------------===//

/// defineBody - When created, RecordDecl's correspond to a forward declared
/// record.  This method is used to mark the decl as being defined, with the
/// specified contents.
void RecordDecl::defineBody(FieldDecl **members, unsigned numMembers) {
  assert(!isDefinition() && "Cannot redefine record!");
  setDefinition(true);
  NumMembers = numMembers;
  if (numMembers) {
    Members = new FieldDecl*[numMembers];
    memcpy(Members, members, numMembers*sizeof(Decl*));
  }
}

FieldDecl *RecordDecl::getMember(IdentifierInfo *II) {
  if (Members == 0 || NumMembers < 0)
    return 0;
  
  // Linear search.  When C++ classes come along, will likely need to revisit.
  for (int i = 0; i != NumMembers; ++i)
    if (Members[i]->getIdentifier() == II)
      return Members[i];
  return 0;
}
