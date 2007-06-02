//===--- CGDecl.cpp - Emit LLVM Code for declarations ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Decl nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "clang/AST/AST.h"
//#include "llvm/Constants.h"
//#include "llvm/DerivedTypes.h"
//#include "llvm/Function.h"
using namespace llvm;
using namespace clang;
using namespace CodeGen;


void CodeGenFunction::EmitDeclStmt(const DeclStmt &S) {
  const Decl *Decl = S.getDecl();

  switch (Decl->getKind()) {
  default: assert(0 && "Unknown decl kind!");
  case Decl::FileVariable:
    assert(0 && "Should not see file-scope variables inside a function!");
  case Decl::ParmVariable:
    assert(0 && "Parmdecls should not be in declstmts!");
  case Decl::Typedef:   // typedef int X;
  case Decl::Function:  // void X();
  case Decl::Struct:    // struct X;
  case Decl::Union:     // union X;
  case Decl::Class:     // class X;
  case Decl::Enum:      // enum X;
    // None of these decls require codegen support.
    return;
    
  case Decl::BlockVariable:
    return EmitBlockVarDecl(cast<BlockVarDecl>(*Decl));
  case Decl::EnumConstant:
    return EmitEnumConstantDecl(cast<EnumConstantDecl>(*Decl));
  }
}

void CodeGenFunction::EmitBlockVarDecl(const BlockVarDecl &D) {
  //assert(0 && "FIXME: Enum constant decls not implemented yet!");  
  
}

void CodeGenFunction::EmitEnumConstantDecl(const EnumConstantDecl &D) {
  assert(0 && "FIXME: Enum constant decls not implemented yet!");  
}
