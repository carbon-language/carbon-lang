//===--- DeclSerialization.cpp - Serialization of Decls ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines methods that implement bitcode serialization for Decls.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "llvm/Bitcode/Serialize.h"
#include "llvm/Bitcode/Deserialize.h"

using namespace clang;

void Decl::Emit(llvm::Serializer& S) const {
  S.EmitInt(getKind());

  switch (getKind()) {
    default:
      assert (false && "Not implemented.");
      break;
      
    case BlockVar:
      cast<BlockVarDecl>(this)->Emit(S);
      break;
      
    case FileVar:
      cast<FileVarDecl>(this)->Emit(S);
      break;
      
    case ParmVar:
      cast<ParmVarDecl>(this)->Emit(S);
      break;
      
    case Function:
      cast<FunctionDecl>(this)->Emit(S);
      break;
  }
}

Decl* Decl::Materialize(llvm::Deserializer& D) {
  assert (false && "FIXME: not implemented.");
  return NULL;
}

void NamedDecl::InternalEmit(llvm::Serializer& S) const {
  S.EmitPtr(Identifier);
}

void NamedDecl::InternalRead(llvm::Deserializer& D) {
  D.ReadPtr(Identifier);
}

void ScopedDecl::InternalEmit(llvm::Serializer& S) const {
  NamedDecl::InternalEmit(S);
  S.EmitPtr(Next);
  S.EmitOwnedPtr<Decl>(NextDeclarator);  
}

void ScopedDecl::InternalRead(llvm::Deserializer& D) {
  NamedDecl::InternalRead(D);
  D.ReadPtr(Next);
  NextDeclarator = cast<ScopedDecl>(D.ReadOwnedPtr<Decl>());
}

void ValueDecl::InternalEmit(llvm::Serializer& S) const {
  S.Emit(DeclType);
  ScopedDecl::InternalEmit(S);
}

void ValueDecl::InternalRead(llvm::Deserializer& D) {
  D.Read(DeclType);
  ScopedDecl::InternalRead(D);
}

void VarDecl::InternalEmit(llvm::Serializer& S) const {
  S.EmitInt(SClass);
  S.EmitInt(objcDeclQualifier);
  VarDecl::InternalEmit(S);
  S.EmitOwnedPtr(Init);
}

void VarDecl::InternalRead(llvm::Deserializer& D) {
  SClass = D.ReadInt();
  objcDeclQualifier = static_cast<ObjcDeclQualifier>(D.ReadInt());
  VarDecl::InternalRead(D);
  D.ReadOwnedPtr(Init);
}


void BlockVarDecl::Emit(llvm::Serializer& S) const {
  S.Emit(getLocation());  
  VarDecl::InternalEmit(S);  
}

BlockVarDecl* BlockVarDecl::Materialize(llvm::Deserializer& D) {
  SourceLocation L = SourceLocation::ReadVal(D);
  BlockVarDecl* decl = new BlockVarDecl(L,NULL,QualType(),None,NULL);
  decl->VarDecl::InternalRead(D);
  return decl;
}

void FileVarDecl::Emit(llvm::Serializer& S) const {
  S.Emit(getLocation());  
  VarDecl::InternalEmit(S);  
}

FileVarDecl* FileVarDecl::Materialize(llvm::Deserializer& D) {
  SourceLocation L = SourceLocation::ReadVal(D);
  FileVarDecl* decl = new FileVarDecl(L,NULL,QualType(),None,NULL);
  decl->VarDecl::InternalRead(D);
  return decl;
}

void ParmVarDecl::Emit(llvm::Serializer& S) const {
  S.Emit(getLocation());  
  VarDecl::InternalEmit(S);  
}

ParmVarDecl* ParmVarDecl::Materialize(llvm::Deserializer& D) {
  SourceLocation L = SourceLocation::ReadVal(D);
  ParmVarDecl* decl = new ParmVarDecl(L,NULL,QualType(),None,NULL);
  decl->VarDecl::InternalRead(D);
  return decl;
}

void FunctionDecl::Emit(llvm::Serializer& S) const {
  S.Emit(getLocation());
  S.EmitInt(SClass);
  S.EmitBool(IsInline);
  
  ValueDecl::InternalEmit(S);

  unsigned NumParams = getNumParams();
  S.EmitInt(NumParams);
  
  for (unsigned i = 0 ; i < NumParams; ++i)
    S.EmitOwnedPtr(ParamInfo[i]);
  
  S.EmitOwnedPtr(Body);
}

FunctionDecl* FunctionDecl::Materialize(llvm::Deserializer& D) {
  SourceLocation L = SourceLocation::ReadVal(D);
  StorageClass SClass = static_cast<StorageClass>(D.ReadInt());
  bool IsInline = D.ReadBool();

  FunctionDecl* decl = new FunctionDecl(L,NULL,QualType(),SClass,IsInline);
  
  decl->ValueDecl::InternalRead(D);

  unsigned NumParams = D.ReadInt();
  decl->ParamInfo = NumParams ? new ParmVarDecl*[NumParams] : NULL;
  
  for (unsigned i = 0 ; i < NumParams; ++i)
    D.ReadOwnedPtr(decl->ParamInfo[i]);

  D.ReadOwnedPtr(decl->Body);
  
  return decl;
}
