//===--- Decl.cpp - Declaration AST Node Implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Decl subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/IdentifierTable.h"

using namespace clang;

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


ImplicitParamDecl *ImplicitParamDecl::Create(ASTContext &C, DeclContext *DC,
    SourceLocation L, IdentifierInfo *Id, QualType T) {
  void *Mem = C.getAllocator().Allocate<ImplicitParamDecl>();
  return new (Mem) ImplicitParamDecl(ImplicitParam, DC, L, Id, T);
}

ParmVarDecl *ParmVarDecl::Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation L, IdentifierInfo *Id,
                                 QualType T, StorageClass S,
                                 Expr *DefArg) {
  void *Mem = C.getAllocator().Allocate<ParmVarDecl>();
  return new (Mem) ParmVarDecl(ParmVar, DC, L, Id, T, S, DefArg);
}

QualType ParmVarDecl::getOriginalType() const {
  if (const ParmVarWithOriginalTypeDecl *PVD = 
      dyn_cast<ParmVarWithOriginalTypeDecl>(this))
    return PVD->OriginalType;
  return getType();
}

ParmVarWithOriginalTypeDecl *ParmVarWithOriginalTypeDecl::Create(
                                 ASTContext &C, DeclContext *DC,
                                 SourceLocation L, IdentifierInfo *Id,
                                 QualType T, QualType OT, StorageClass S,
                                 Expr *DefArg) {
  void *Mem = C.getAllocator().Allocate<ParmVarWithOriginalTypeDecl>();
  return new (Mem) ParmVarWithOriginalTypeDecl(DC, L, Id, T, OT, S, DefArg);
}

FunctionDecl *FunctionDecl::Create(ASTContext &C, DeclContext *DC,
                                   SourceLocation L, 
                                   DeclarationName N, QualType T, 
                                   StorageClass S, bool isInline, 
                                   SourceLocation TypeSpecStartLoc) {
  void *Mem = C.getAllocator().Allocate<FunctionDecl>();
  return new (Mem) FunctionDecl(Function, DC, L, N, T, S, isInline,
                                TypeSpecStartLoc);
}

BlockDecl *BlockDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L) {
  void *Mem = C.getAllocator().Allocate<BlockDecl>();
  return new (Mem) BlockDecl(DC, L);
}

FieldDecl *FieldDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                             IdentifierInfo *Id, QualType T, Expr *BW,
                             bool Mutable) {
  void *Mem = C.getAllocator().Allocate<FieldDecl>();
  return new (Mem) FieldDecl(Decl::Field, DC, L, Id, T, BW, Mutable);
}

bool FieldDecl::isAnonymousStructOrUnion() const {
  if (!isImplicit() || getDeclName())
    return false;
  
  if (const RecordType *Record = getType()->getAsRecordType())
    return Record->getDecl()->isAnonymousStructOrUnion();

  return false;
}

EnumConstantDecl *EnumConstantDecl::Create(ASTContext &C, EnumDecl *CD,
                                           SourceLocation L,
                                           IdentifierInfo *Id, QualType T,
                                           Expr *E, const llvm::APSInt &V) {
  void *Mem = C.getAllocator().Allocate<EnumConstantDecl>();
  return new (Mem) EnumConstantDecl(CD, L, Id, T, E, V);
}

void EnumConstantDecl::Destroy(ASTContext& C) {
  if (Init) Init->Destroy(C);
  Decl::Destroy(C);
}

TypedefDecl *TypedefDecl::Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation L,
                                 IdentifierInfo *Id, QualType T) {
  void *Mem = C.getAllocator().Allocate<TypedefDecl>();
  return new (Mem) TypedefDecl(DC, L, Id, T);
}

EnumDecl *EnumDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                           IdentifierInfo *Id,
                           EnumDecl *PrevDecl) {
  void *Mem = C.getAllocator().Allocate<EnumDecl>();
  EnumDecl *Enum = new (Mem) EnumDecl(DC, L, Id);
  C.getTypeDeclType(Enum, PrevDecl);
  return Enum;
}

void EnumDecl::Destroy(ASTContext& C) {
  Decl::Destroy(C);
}

void EnumDecl::completeDefinition(ASTContext &C, QualType NewType) {
  assert(!isDefinition() && "Cannot redefine enums!");
  IntegerType = NewType;
  TagDecl::completeDefinition();
}

FileScopeAsmDecl *FileScopeAsmDecl::Create(ASTContext &C, DeclContext *DC,
                                           SourceLocation L,
                                           StringLiteral *Str) {
  void *Mem = C.getAllocator().Allocate<FileScopeAsmDecl>();
  return new (Mem) FileScopeAsmDecl(DC, L, Str);
}

//===----------------------------------------------------------------------===//
// NamedDecl Implementation
//===----------------------------------------------------------------------===//

bool NamedDecl::declarationReplaces(NamedDecl *OldD) const {
  assert(getDeclName() == OldD->getDeclName() && "Declaration name mismatch");

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(this))
    // For function declarations, we keep track of redeclarations.
    return FD->getPreviousDeclaration() == OldD;

  // For non-function declarations, if the declarations are of the
  // same kind then this must be a redeclaration, or semantic analysis
  // would not have given us the new declaration.
  return this->getKind() == OldD->getKind();
}


//===----------------------------------------------------------------------===//
// VarDecl Implementation
//===----------------------------------------------------------------------===//

VarDecl *VarDecl::Create(ASTContext &C, DeclContext *DC, SourceLocation L,
                         IdentifierInfo *Id, QualType T, StorageClass S, 
                         SourceLocation TypeSpecStartLoc) {
  void *Mem = C.getAllocator().Allocate<VarDecl>();
  return new (Mem) VarDecl(Var, DC, L, Id, T, S, TypeSpecStartLoc);
}

void VarDecl::Destroy(ASTContext& C) {
  this->~VarDecl();
  C.getAllocator().Deallocate((void *)this);
}

VarDecl::~VarDecl() {
  delete getInit();
}

//===----------------------------------------------------------------------===//
// FunctionDecl Implementation
//===----------------------------------------------------------------------===//

void FunctionDecl::Destroy(ASTContext& C) {
  if (Body)
    Body->Destroy(C);

  for (param_iterator I=param_begin(), E=param_end(); I!=E; ++I)
    (*I)->Destroy(C);

  C.getAllocator().Deallocate(ParamInfo);

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

// Helper function for FunctionDecl::getNumParams and FunctionDecl::setParams()
static unsigned getNumTypeParams(QualType T) {
  const FunctionType *FT = T->getAsFunctionType();
  if (isa<FunctionTypeNoProto>(FT))
    return 0;
  return cast<FunctionTypeProto>(FT)->getNumArgs();
}

unsigned FunctionDecl::getNumParams() const {
  // Can happen if a FunctionDecl is declared using typeof(some_other_func) bar;
  if (!ParamInfo)
    return 0;
  
  return getNumTypeParams(getType());
}

void FunctionDecl::setParams(ASTContext& C, ParmVarDecl **NewParamInfo,
                             unsigned NumParams) {
  assert(ParamInfo == 0 && "Already has param info!");
  assert(NumParams == getNumTypeParams(getType()) &&
         "Parameter count mismatch!");
  
  // Zero params -> null pointer.
  if (NumParams) {
    void *Mem = C.getAllocator().Allocate<ParmVarDecl*>(NumParams);
    ParamInfo = new (Mem) ParmVarDecl*[NumParams];
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

/// getOverloadedOperator - Which C++ overloaded operator this
/// function represents, if any.
OverloadedOperatorKind FunctionDecl::getOverloadedOperator() const {
  if (getDeclName().getNameKind() == DeclarationName::CXXOperatorName)
    return getDeclName().getCXXOverloadedOperator();
  else
    return OO_None;
}

//===----------------------------------------------------------------------===//
// TagDecl Implementation
//===----------------------------------------------------------------------===//

void TagDecl::startDefinition() {
  cast<TagType>(TypeForDecl)->decl.setPointer(this);
  cast<TagType>(TypeForDecl)->decl.setInt(1);
}

void TagDecl::completeDefinition() {
  assert((!TypeForDecl || 
          cast<TagType>(TypeForDecl)->decl.getPointer() == this) &&
         "Attempt to redefine a tag definition?");
  IsDefinition = true;
  cast<TagType>(TypeForDecl)->decl.setPointer(this);
  cast<TagType>(TypeForDecl)->decl.setInt(0);
}

TagDecl* TagDecl::getDefinition(ASTContext& C) const {
  QualType T = C.getTypeDeclType(const_cast<TagDecl*>(this));
  TagDecl* D = cast<TagDecl>(cast<TagType>(T)->getDecl());  
  return D->isDefinition() ? D : 0;
}

//===----------------------------------------------------------------------===//
// RecordDecl Implementation
//===----------------------------------------------------------------------===//

RecordDecl::RecordDecl(Kind DK, TagKind TK, DeclContext *DC, SourceLocation L,
                       IdentifierInfo *Id)
  : TagDecl(DK, TK, DC, L, Id) {
  HasFlexibleArrayMember = false;
  AnonymousStructOrUnion = false;
  assert(classof(static_cast<Decl*>(this)) && "Invalid Kind!");
}

RecordDecl *RecordDecl::Create(ASTContext &C, TagKind TK, DeclContext *DC,
                               SourceLocation L, IdentifierInfo *Id,
                               RecordDecl* PrevDecl) {
  
  void *Mem = C.getAllocator().Allocate<RecordDecl>();
  RecordDecl* R = new (Mem) RecordDecl(Record, TK, DC, L, Id);
  C.getTypeDeclType(R, PrevDecl);
  return R;
}

RecordDecl::~RecordDecl() {
}

void RecordDecl::Destroy(ASTContext& C) {
  DeclContext::DestroyDecls(C);
  TagDecl::Destroy(C);
}

/// completeDefinition - Notes that the definition of this type is now
/// complete.
void RecordDecl::completeDefinition(ASTContext& C) {
  assert(!isDefinition() && "Cannot redefine record!");
  TagDecl::completeDefinition();
}

//===----------------------------------------------------------------------===//
// BlockDecl Implementation
//===----------------------------------------------------------------------===//

BlockDecl::~BlockDecl() {
}

void BlockDecl::Destroy(ASTContext& C) {
  if (Body)
    Body->Destroy(C);

  for (param_iterator I=param_begin(), E=param_end(); I!=E; ++I)
    (*I)->Destroy(C);
    
  Decl::Destroy(C);
}
