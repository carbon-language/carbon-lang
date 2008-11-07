//===--- DeclSerialization.cpp - Serialization of Decls ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines methods that implement bitcode serialization for Decls.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"

using llvm::Serializer;
using llvm::Deserializer;
using llvm::SerializedPtrID;

using namespace clang;

//===----------------------------------------------------------------------===//
// Decl Serialization: Dispatch code to handle specialized decl types.
//===----------------------------------------------------------------------===//

void Decl::Emit(Serializer& S) const {
  S.EmitInt(getKind());
  EmitImpl(S);
  if (const DeclContext *DC = dyn_cast<const DeclContext>(this))
    DC->EmitOutRec(S);
}

Decl* Decl::Create(Deserializer& D, ASTContext& C) {

  Decl *Dcl;
  Kind k = static_cast<Kind>(D.ReadInt());

  switch (k) {
    default:
      assert (false && "Not implemented.");

    case TranslationUnit:
      Dcl = TranslationUnitDecl::CreateImpl(D, C);
      break;

    case Namespace:
      Dcl = NamespaceDecl::CreateImpl(D, C);
      break;

    case Var:
      Dcl = VarDecl::CreateImpl(D, C);
      break;
      
    case Enum:
      Dcl = EnumDecl::CreateImpl(D, C);
      break;
      
    case EnumConstant:
      Dcl = EnumConstantDecl::CreateImpl(D, C);
      break;
      
    case Field:
      Dcl = FieldDecl::CreateImpl(D, C);
      break;
      
    case ParmVar:
      Dcl = ParmVarDecl::CreateImpl(D, C);
      break;
      
    case Function:
      Dcl = FunctionDecl::CreateImpl(D, C);
      break;

    case OverloadedFunction:
      Dcl = OverloadedFunctionDecl::CreateImpl(D, C);
      break;

    case Record:
      Dcl = RecordDecl::CreateImpl(D, C);
      break;
      
    case Typedef:
      Dcl = TypedefDecl::CreateImpl(D, C);
      break;
      
    case FileScopeAsm:
      Dcl = FileScopeAsmDecl::CreateImpl(D, C);
      break;
  }

  if (DeclContext *DC = dyn_cast<DeclContext>(Dcl))
    DC->ReadOutRec(D, C);

  return Dcl;
}

//===----------------------------------------------------------------------===//
//      Common serialization logic for subclasses of Decl.
//===----------------------------------------------------------------------===//

void Decl::EmitInRec(Serializer& S) const {
  S.Emit(getLocation());                    // From Decl.
}

void Decl::ReadInRec(Deserializer& D, ASTContext& C) {
  Loc = SourceLocation::ReadVal(D);                 // From Decl.
}

//===----------------------------------------------------------------------===//
//      Common serialization logic for subclasses of DeclContext.
//===----------------------------------------------------------------------===//

void DeclContext::EmitOutRec(Serializer& S) const {
  S.EmitPtr(DeclChain);
}

void DeclContext::ReadOutRec(Deserializer& D, ASTContext& C) {
  D.ReadPtr(DeclChain);
}

//===----------------------------------------------------------------------===//
//      Common serialization logic for subclasses of NamedDecl.
//===----------------------------------------------------------------------===//

void NamedDecl::EmitInRec(Serializer& S) const {
  Decl::EmitInRec(S);
  S.EmitPtr(getIdentifier());               // From NamedDecl.
}

void NamedDecl::ReadInRec(Deserializer& D, ASTContext& C) {
  Decl::ReadInRec(D, C);
  D.ReadPtr(Identifier);                            // From NamedDecl.  
}

//===----------------------------------------------------------------------===//
//      Common serialization logic for subclasses of ScopedDecl.
//===----------------------------------------------------------------------===//

void ScopedDecl::EmitInRec(Serializer& S) const {
  NamedDecl::EmitInRec(S);
  S.EmitPtr(getNext());                     // From ScopedDecl.  
  S.EmitPtr(cast_or_null<Decl>(getDeclContext()));  // From ScopedDecl.
}

void ScopedDecl::ReadInRec(Deserializer& D, ASTContext& C) {
  NamedDecl::ReadInRec(D, C);
  D.ReadPtr(Next);                                  // From ScopedDecl.
  
  assert(DeclCtx == 0);   // Allow back-patching.  Observe that we register
  D.ReadPtr(DeclCtx);     // the variable of the *object* for back-patching.
                          // Its actual value will get filled in later.
}
    
  //===------------------------------------------------------------===//
  // NOTE: Not all subclasses of ScopedDecl will use the "OutRec"     //
  //   methods.  This is because owned pointers are usually "batched" //
  //   together for efficiency.                                       //
  //===------------------------------------------------------------===//

void ScopedDecl::EmitOutRec(Serializer& S) const {
  S.EmitOwnedPtr(getNextDeclarator());   // From ScopedDecl.
}

void ScopedDecl::ReadOutRec(Deserializer& D, ASTContext& C) {
  NextDeclarator = 
    cast_or_null<ScopedDecl>(D.ReadOwnedPtr<Decl>(C)); // From ScopedDecl.
}

//===----------------------------------------------------------------------===//
//      Common serialization logic for subclasses of ValueDecl.
//===----------------------------------------------------------------------===//

void ValueDecl::EmitInRec(Serializer& S) const {
  ScopedDecl::EmitInRec(S);
  S.Emit(getType());                        // From ValueDecl.
}

void ValueDecl::ReadInRec(Deserializer& D, ASTContext& C) {
  ScopedDecl::ReadInRec(D, C);
  DeclType = QualType::ReadVal(D);          // From ValueDecl.
}

//===----------------------------------------------------------------------===//
//      Common serialization logic for subclasses of VarDecl.
//===----------------------------------------------------------------------===//

void VarDecl::EmitInRec(Serializer& S) const {
  ValueDecl::EmitInRec(S);
  S.EmitInt(getStorageClass());             // From VarDecl.
}

void VarDecl::ReadInRec(Deserializer& D, ASTContext& C) {
  ValueDecl::ReadInRec(D, C);
  SClass = static_cast<StorageClass>(D.ReadInt());  // From VarDecl. 
}

    //===------------------------------------------------------------===//
    // NOTE: VarDecl has its own "OutRec" methods that doesn't use      //
    //  the one define in ScopedDecl.  This is to batch emit the        //
    //  owned pointers, which results in a smaller output.
    //===------------------------------------------------------------===//

void VarDecl::EmitOutRec(Serializer& S) const {
  // Emit these last because they will create records of their own.
  S.BatchEmitOwnedPtrs(getInit(),            // From VarDecl.
                       getNextDeclarator()); // From ScopedDecl.  
}

void VarDecl::ReadOutRec(Deserializer& D, ASTContext& C) {
  Decl* next_declarator;
  
  D.BatchReadOwnedPtrs(Init,             // From VarDecl.
                       next_declarator,  // From ScopedDecl.
                       C);
  
  setNextDeclarator(cast_or_null<ScopedDecl>(next_declarator));
}


void VarDecl::EmitImpl(Serializer& S) const {
  VarDecl::EmitInRec(S);
  VarDecl::EmitOutRec(S);
}

void VarDecl::ReadImpl(Deserializer& D, ASTContext& C) {
  ReadInRec(D, C);
  ReadOutRec(D, C);
}

//===----------------------------------------------------------------------===//
//      TranslationUnitDecl Serialization.
//===----------------------------------------------------------------------===//

void TranslationUnitDecl::EmitImpl(llvm::Serializer& S) const
{
  Decl::EmitInRec(S);
}

TranslationUnitDecl* TranslationUnitDecl::CreateImpl(Deserializer& D,
                                                     ASTContext& C) {  
  void *Mem = C.getAllocator().Allocate<TranslationUnitDecl>();
  TranslationUnitDecl* decl = new (Mem) TranslationUnitDecl();
 
  decl->Decl::ReadInRec(D, C);
  
  return decl;
}

//===----------------------------------------------------------------------===//
//      NamespaceDecl Serialization.
//===----------------------------------------------------------------------===//

void NamespaceDecl::EmitImpl(llvm::Serializer& S) const
{
  ScopedDecl::EmitInRec(S);
  S.Emit(getLBracLoc());
  S.Emit(getRBracLoc());
  ScopedDecl::EmitOutRec(S);
}

NamespaceDecl* NamespaceDecl::CreateImpl(Deserializer& D, ASTContext& C) {  
  void *Mem = C.getAllocator().Allocate<NamespaceDecl>();
  NamespaceDecl* decl = new (Mem) NamespaceDecl(0, SourceLocation(), 0);
 
  decl->ScopedDecl::ReadInRec(D, C);
  decl->LBracLoc = SourceLocation::ReadVal(D);
  decl->RBracLoc = SourceLocation::ReadVal(D);
  decl->ScopedDecl::ReadOutRec(D, C);
  
  return decl;
}

//===----------------------------------------------------------------------===//
//      VarDecl Serialization.
//===----------------------------------------------------------------------===//

VarDecl* VarDecl::CreateImpl(Deserializer& D, ASTContext& C) {  
  void *Mem = C.getAllocator().Allocate<VarDecl>();
  VarDecl* decl =
    new (Mem) VarDecl(Var, 0, SourceLocation(), NULL, QualType(), None, NULL);
 
  decl->VarDecl::ReadImpl(D, C);
  return decl;
}

//===----------------------------------------------------------------------===//
//      ParmVarDecl Serialization.
//===----------------------------------------------------------------------===//

void ParmVarDecl::EmitImpl(llvm::Serializer& S) const {
  VarDecl::EmitImpl(S);
  S.EmitInt(getObjCDeclQualifier());        // From ParmVarDecl.
  S.EmitOwnedPtr(getDefaultArg());          // From ParmVarDecl.
}

ParmVarDecl* ParmVarDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  void *Mem = C.getAllocator().Allocate<ParmVarDecl>();
  ParmVarDecl* decl = new (Mem)
    ParmVarDecl(0, SourceLocation(), NULL, QualType(), None, NULL, NULL);
  
  decl->VarDecl::ReadImpl(D, C);
  decl->objcDeclQualifier = static_cast<ObjCDeclQualifier>(D.ReadInt());
  decl->DefaultArg = D.ReadOwnedPtr<Expr>(C);
  return decl;
}

//===----------------------------------------------------------------------===//
//      EnumDecl Serialization.
//===----------------------------------------------------------------------===//

void EnumDecl::EmitImpl(Serializer& S) const {
  ScopedDecl::EmitInRec(S);
  S.EmitBool(isDefinition());
  S.Emit(IntegerType);  
  S.BatchEmitOwnedPtrs(getEnumConstantList(),getNextDeclarator());
}

EnumDecl* EnumDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  void *Mem = C.getAllocator().Allocate<EnumDecl>();
  EnumDecl* decl = new (Mem) EnumDecl(0, SourceLocation(), NULL, NULL);
  
  decl->ScopedDecl::ReadInRec(D, C);
  decl->setDefinition(D.ReadBool());
  decl->IntegerType = QualType::ReadVal(D);
  
  Decl* next_declarator;
  Decl* Elist;
  
  D.BatchReadOwnedPtrs(Elist, next_declarator, C);
  
  decl->setDeclChain(cast_or_null<EnumConstantDecl>(Elist));
  decl->setNextDeclarator(cast_or_null<ScopedDecl>(next_declarator));
  
  return decl;
}

//===----------------------------------------------------------------------===//
//      EnumConstantDecl Serialization.
//===----------------------------------------------------------------------===//

void EnumConstantDecl::EmitImpl(Serializer& S) const {
  S.Emit(Val);
  ValueDecl::EmitInRec(S);
  S.BatchEmitOwnedPtrs(getNextDeclarator(),Init);
}
 
EnumConstantDecl* EnumConstantDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  llvm::APSInt val(1);
  D.Read(val);
  
  void *Mem = C.getAllocator().Allocate<EnumConstantDecl>();
  EnumConstantDecl* decl = new (Mem)
    EnumConstantDecl(0, SourceLocation(), NULL, QualType(), NULL, val, NULL);
  
  decl->ValueDecl::ReadInRec(D, C);
  
  Decl* next_declarator;
  
  D.BatchReadOwnedPtrs(next_declarator, decl->Init, C);
  
  decl->setNextDeclarator(cast_or_null<ScopedDecl>(next_declarator));

  return decl;    
}

//===----------------------------------------------------------------------===//
//      FieldDecl Serialization.
//===----------------------------------------------------------------------===//

void FieldDecl::EmitImpl(Serializer& S) const {
  S.Emit(getType());
  NamedDecl::EmitInRec(S);
  S.EmitOwnedPtr(BitWidth);  
}

FieldDecl* FieldDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  void *Mem = C.getAllocator().Allocate<FieldDecl>();
  FieldDecl* decl = new (Mem) FieldDecl(SourceLocation(), NULL, QualType(), 0);
  decl->DeclType.ReadBackpatch(D);  
  decl->ReadInRec(D, C);
  decl->BitWidth = D.ReadOwnedPtr<Expr>(C);
  return decl;
}

//===----------------------------------------------------------------------===//
//      FunctionDecl Serialization.
//===----------------------------------------------------------------------===//

void FunctionDecl::EmitImpl(Serializer& S) const {
  S.EmitInt(SClass);           // From FunctionDecl.
  S.EmitBool(IsInline);        // From FunctionDecl.
  ValueDecl::EmitInRec(S);
  S.EmitPtr(PreviousDeclaration);
  
  // NOTE: We do not need to serialize out the number of parameters, because
  //  that is encoded in the type (accessed via getNumParams()).
  
  if (ParamInfo != NULL) {
    S.EmitBool(true);
    S.EmitInt(getNumParams());
    S.BatchEmitOwnedPtrs(getNumParams(),&ParamInfo[0], Body,
                         getNextDeclarator());
  }
  else {
    S.EmitBool(false);
    S.BatchEmitOwnedPtrs(Body,getNextDeclarator());  
  }
}

FunctionDecl* FunctionDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  StorageClass SClass = static_cast<StorageClass>(D.ReadInt());
  bool IsInline = D.ReadBool();
  
  void *Mem = C.getAllocator().Allocate<FunctionDecl>();
  FunctionDecl* decl = new (Mem)
    FunctionDecl(Function, 0, SourceLocation(), NULL,
                 QualType(), SClass, IsInline, 0);
  
  decl->ValueDecl::ReadInRec(D, C);
  D.ReadPtr(decl->PreviousDeclaration);

  Decl* next_declarator;
  
  int numParams;
  bool hasParamDecls = D.ReadBool();
  if (hasParamDecls)
    numParams = D.ReadInt();
    
  decl->ParamInfo = hasParamDecls
                  ? new ParmVarDecl*[numParams] 
                  : NULL;  
  
  if (hasParamDecls)
    D.BatchReadOwnedPtrs(numParams,
                         reinterpret_cast<Decl**>(&decl->ParamInfo[0]),
                         decl->Body, next_declarator, C);
  else
    D.BatchReadOwnedPtrs(decl->Body, next_declarator, C);
  
  decl->setNextDeclarator(cast_or_null<ScopedDecl>(next_declarator));
  
  return decl;
}

void BlockDecl::EmitImpl(Serializer& S) const {
  // FIXME: what about arguments?
  S.Emit(getCaretLocation());
  S.EmitOwnedPtr(Body);
}

BlockDecl* BlockDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  QualType Q = QualType::ReadVal(D);
  SourceLocation L = SourceLocation::ReadVal(D);
  /*CompoundStmt* BodyStmt = cast<CompoundStmt>(*/D.ReadOwnedPtr<Stmt>(C)/*)*/;
  assert(0 && "Cannot deserialize BlockBlockExpr yet");
  // FIXME: need to handle parameters.
  //return new BlockBlockExpr(L, Q, BodyStmt);
  return 0;
}

//===----------------------------------------------------------------------===//
//      OverloadedFunctionDecl Serialization.
//===----------------------------------------------------------------------===//

void OverloadedFunctionDecl::EmitImpl(Serializer& S) const {
  NamedDecl::EmitInRec(S);

  S.EmitInt(getNumFunctions());
  for (unsigned func = 0; func < getNumFunctions(); ++func)
    S.EmitPtr(Functions[func]);
}

OverloadedFunctionDecl * 
OverloadedFunctionDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  void *Mem = C.getAllocator().Allocate<OverloadedFunctionDecl>();
  OverloadedFunctionDecl* decl = new (Mem)
    OverloadedFunctionDecl(0, NULL);
  
  decl->NamedDecl::ReadInRec(D, C);

  unsigned numFunctions = D.ReadInt();
  decl->Functions.reserve(numFunctions);
  for (unsigned func = 0; func < numFunctions; ++func)
    D.ReadPtr(decl->Functions[func]);
  
  return decl;
}

//===----------------------------------------------------------------------===//
//      RecordDecl Serialization.
//===----------------------------------------------------------------------===//

void RecordDecl::EmitImpl(Serializer& S) const {
  S.EmitInt(getTagKind());

  ScopedDecl::EmitInRec(S);
  S.EmitBool(isDefinition());
  S.EmitBool(hasFlexibleArrayMember());
  S.EmitSInt(getNumMembers());
  if (getNumMembers() > 0) {
    assert (Members);
    S.BatchEmitOwnedPtrs((unsigned) getNumMembers(), (Decl**) &Members[0]);
  }
  else
    ScopedDecl::EmitOutRec(S);
}

RecordDecl* RecordDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  TagKind TK = TagKind(D.ReadInt());

  void *Mem = C.getAllocator().Allocate<RecordDecl>();
  RecordDecl* decl = new (Mem) RecordDecl(Record, TK, 0, SourceLocation(), NULL);
    
  decl->ScopedDecl::ReadInRec(D, C);
  decl->setDefinition(D.ReadBool());
  decl->setHasFlexibleArrayMember(D.ReadBool());
  decl->NumMembers = D.ReadSInt();
  
  if (decl->getNumMembers() > 0) {
    decl->Members = new FieldDecl*[(unsigned) decl->getNumMembers()];
                              
    D.BatchReadOwnedPtrs((unsigned) decl->getNumMembers(),
                         (Decl**) &decl->Members[0], C);
  }
  else
    decl->ScopedDecl::ReadOutRec(D, C);
  
  return decl;
}

//===----------------------------------------------------------------------===//
//      TypedefDecl Serialization.
//===----------------------------------------------------------------------===//

void TypedefDecl::EmitImpl(Serializer& S) const {
  S.Emit(UnderlyingType);
  ScopedDecl::EmitInRec(S);
  ScopedDecl::EmitOutRec(S);
}

TypedefDecl* TypedefDecl::CreateImpl(Deserializer& D, ASTContext& C) {
  QualType T = QualType::ReadVal(D);
  
  void *Mem = C.getAllocator().Allocate<TypedefDecl>();
  TypedefDecl* decl = new (Mem) TypedefDecl(0, SourceLocation(), NULL, T, NULL);
  
  decl->ScopedDecl::ReadInRec(D, C);
  decl->ScopedDecl::ReadOutRec(D, C);

  return decl;
}

//===----------------------------------------------------------------------===//
//      LinkageSpec Serialization.
//===----------------------------------------------------------------------===//

void LinkageSpecDecl::EmitInRec(Serializer& S) const {
  Decl::EmitInRec(S);
  S.EmitInt(getLanguage());
  S.EmitPtr(D);
}

void LinkageSpecDecl::ReadInRec(Deserializer& D, ASTContext& C) {
  Decl::ReadInRec(D, C);
  Language = static_cast<LanguageIDs>(D.ReadInt());
  D.ReadPtr(this->D);
}

//===----------------------------------------------------------------------===//
//      FileScopeAsm Serialization.
//===----------------------------------------------------------------------===//

void FileScopeAsmDecl::EmitImpl(llvm::Serializer& S) const
{
  Decl::EmitInRec(S);
  S.EmitOwnedPtr(AsmString);
}

FileScopeAsmDecl* FileScopeAsmDecl::CreateImpl(Deserializer& D, ASTContext& C) { 
  void *Mem = C.getAllocator().Allocate<FileScopeAsmDecl>();
  FileScopeAsmDecl* decl = new (Mem) FileScopeAsmDecl(SourceLocation(), 0);

  decl->Decl::ReadInRec(D, C);
  decl->AsmString = cast<StringLiteral>(D.ReadOwnedPtr<Expr>(C));
//  D.ReadOwnedPtr(D.ReadOwnedPtr<StringLiteral>())<#T * * Ptr#>, <#bool AutoRegister#>)(decl->AsmString);
  
  return decl;
}
