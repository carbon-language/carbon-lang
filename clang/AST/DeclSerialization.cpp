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

using llvm::SerializeTrait;
using llvm::Deserializer;
using llvm::Serializer;
using namespace clang;


static void EmitEnumConstantDecl(Serializer& S, EnumConstantDecl& decl) {
  S.Emit(decl.getLocation());
  S.EmitPtr(decl.getIdentifier());
//  S.Emit(decl.getType());  FIXME
  S.EmitOwnedPtr<Stmt>(decl.getInitExpr());
    // S.Emit(decl.getInitVal()); FIXME
  S.EmitOwnedPtr<Decl>(decl.getNextDeclarator());
}

static void EmitFunctionDecl(Serializer& S, FunctionDecl& decl) {
  S.Emit(decl.getLocation());
  S.EmitPtr(decl.getIdentifier());
//  S.Emit(decl.getType()); FIXME
//  S.Emit(decl.getStorageClass()); FIXME
  S.EmitBool(decl.isInline());
  S.EmitOwnedPtr<Decl>(decl.getNextDeclarator());
}


void SerializeTrait<Decl>::Emit(Serializer& S, Decl& decl) {
  assert (!decl.isInvalidDecl() && "Can only serialize valid decls.");

  S.EmitInt((unsigned) decl.getKind());

  switch (decl.getKind()) {
    default:
      assert (false && "Serialization not implemented for decl type.");
      return;
      
    case Decl::EnumConstant:
      EmitEnumConstantDecl(S,cast<EnumConstantDecl>(decl));
      return;
      
    case Decl::Function:
      EmitFunctionDecl(S,cast<FunctionDecl>(decl));
      return;      
  }  
}
