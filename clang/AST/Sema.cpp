//===--- Sema.cpp - AST Builder and Semantic Analysis Implementation ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the actions class which performs semantic analysis and
// builds an AST out of a parse stream.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/Preprocessor.h"
using namespace llvm;
using namespace clang;

Sema::Sema(Preprocessor &pp, std::vector<Decl*> &prevInGroup)
  : PP(pp), 
    Context(*new ASTContext(pp.getTargetInfo(), pp.getIdentifierTable())),
    CurFunctionDecl(0), LastInGroupList(prevInGroup) {
}

void Sema::PrintStats() { 
  Context.PrintStats(); 
}

//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

void Sema::Diag(SourceLocation Loc, unsigned DiagID, const std::string &Msg) {
  PP.Diag(Loc, DiagID, Msg);
}

const LangOptions &Sema::getLangOptions() const {
  return PP.getLangOptions();
}
