//===--- Decl.cpp - Declaration AST Node Implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Decl class and subclasses.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "clang/Lex/IdentifierTable.h"
using namespace llvm;
using namespace clang;

// Out-of-line virtual method providing a home for Decl.
Decl::~Decl() {
}

const char *Decl::getName() const {
  return getIdentifier()->getName();
}


FunctionDecl::~FunctionDecl() {
  delete[] ParamInfo;
}

unsigned FunctionDecl::getNumParams() const {
  return cast<FunctionTypeProto>(getType().getTypePtr())->getNumArgs();
}

void FunctionDecl::setParams(VarDecl **NewParamInfo, unsigned NumParams) {
  assert(ParamInfo == 0 && "Already has param info!");
  assert(NumParams == getNumParams() && "Parameter count mismatch!");
  
  ParamInfo = new VarDecl*[NumParams];
  memcpy(ParamInfo, NewParamInfo, sizeof(VarDecl*)*NumParams);
}
